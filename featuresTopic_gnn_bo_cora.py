#!/usr/bin/env python
# coding: utf-8

import sys
import superneuromat as snm
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import yaml
import pickle
from multiprocessing import Pool

import copy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf
import os
import pickle
from skopt import Optimizer, Space
from skopt.space import Categorical, Real
import time

# project specific pip imports
from gnn_citation_networks import GraphData, test_paper
from utils import (
    parse_args,
    print_introduction,
    validate_overrides,
)

class GraphData():
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.paper_to_topic = {} # maps the paper ID in the dataset to its topic ID
        self.index_to_paper = []    # creates an index for each paper
        self.topics = []            # the list of topics
        self.train_papers = []
        self.validation_papers = []
        self.test_papers = []
        self.load_topics()
        if self.name != "microseer":
            self.train_val_test_split()
        else:
            self.train_val_test_split_small()
        self.load_features()
        self.load_graph()


    def load_topics(self):
        if (self.name == "cora"):
            f = open("data/Cora/group-edges.csv", 'r')
            lines = f.readlines()
            f.close()

            for line in lines:
                fields = line.strip().split(",")
                if (fields[1] not in self.topics):
                    self.topics.append(fields[1])
                self.paper_to_topic[fields[0]] = fields[1]
                self.index_to_paper.append(fields[0])
        elif (self.name == "citeseer"):
            f = open("data/citeseer/citeseer.content", 'r')
            lines = f.readlines()
            f.close()

            for line in lines:
                fields = line.strip().split()
                if (fields[-1] not in self.topics):
                    self.topics.append(fields[-1])
                #print(fields[0])
                self.paper_to_topic[fields[0]] = fields[-1]
                self.index_to_paper.append(fields[0])
        elif (self.name == "microseer"):
            f = open("data/microseer/microseer.content", 'r')
            lines = f.readlines()
            f.close()

            for line in lines:
                fields = line.strip().split()
                if (fields[-1] not in self.topics):
                    self.topics.append(fields[-1])
                #print(fields[0])
                self.paper_to_topic[fields[0]] = fields[-1]
                self.index_to_paper.append(fields[0])
        elif (self.name == "miniseer"):
            f = open("data/miniseer/miniseer.content", 'r')
            lines = f.readlines()
            f.close()

            for line in lines:
                fields = line.strip().split()
                if (fields[-1] not in self.topics):
                    self.topics.append(fields[-1])
                #print(fields[0])
                self.paper_to_topic[fields[0]] = fields[-1]
                self.index_to_paper.append(fields[0])
        elif (self.name == "pubmed"):
            f = open("data/Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab", 'r')
            lines = f.readlines()
            f.close()
            lines = lines[2:]

            for line in lines:
                fields = line.strip().split()
                if (fields[1] not in self.topics):
                    self.topics.append(fields[1])
                self.paper_to_topic["paper:"+fields[0]] = fields[1]
                self.index_to_paper.append("paper:"+fields[0])

    def load_features(self):
        self.features = {} # keyed on paper ID, value is the feature vector
        if (self.name == "cora"):
            f = open("data/Cora/cora/cora.content", 'r')
            lines = f.readlines()
            f.close()

            for line in lines:
                fields = line.strip().split()
                paper_id = fields[0]
                feature = [int(x) for x in fields[1:-1]]
                self.features[paper_id] = feature
                self.num_features = len(feature)
            print("NUM PAPERS WITH FEATURES: ", len(self.features.keys()))
        elif (self.name == "citeseer"):
            f = open("data/citeseer/citeseer.content", 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                fields = line.strip().split()
                paper_id = fields[0]
                feature = [int(x) for x in fields[1:-1]]
                self.features[paper_id] = feature
                self.num_features = len(feature)
            print("NUM PAPERS WITH FEATURES: ", len(self.features.keys()))
        elif (self.name == "microseer"):
            f = open("data/microseer/microseer.content", 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                fields = line.strip().split()
                paper_id = fields[0]
                feature = [int(x) for x in fields[1:-1]]
                self.features[paper_id] = feature
                self.num_features = len(feature)
            print("NUM PAPERS WITH FEATURES: ", len(self.features.keys()))
        elif (self.name == "miniseer"):
            f = open("data/miniseer/miniseer.content", 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                fields = line.strip().split()
                paper_id = fields[0]
                feature = [int(x) for x in fields[1:-1]]
                self.features[paper_id] = feature
                self.num_features = len(feature)
            print("NUM PAPERS WITH FEATURES: ", len(self.features.keys()))
        elif (self.name == "pubmed"):
            f = open("data/Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab", 'r')
            lines = f.readlines()
            f.close()
            feature_line = lines[1]
            fields = feature_line.split()
            fields = fields[1:-1]
            all_features = {}
            for i in range(len(fields)):
                feat = fields[i].split(':')[1]
                all_features[feat] = i

            self.num_features = len(all_features.keys())
            lines = lines[2:]
            for line in lines:
                feature = [0]*self.num_features
                fields = line.split()
                paper_id = "paper:" + fields[0]
                xfields = fields[-1].split('=')
                xfields = xfields[1].split(',')
                for x in xfields:
                    feature[all_features[x]] = 1
                self.features[paper_id] = feature

        self.paper_to_features = {} # keyed on paper ID, value is the number of features it has
        self.feature_to_papers = {} # keyed on feature ID, value is the number of papers that have that feature
        for p in self.features.keys():
            self.paper_to_features[p] = np.sum(self.features[p])
            for i in range(len(self.features[p])):
                if (i not in self.feature_to_papers.keys()):
                    self.feature_to_papers[i] = 0
                self.feature_to_papers[i] += self.features[p][i]


    def load_graph(self):
        if (self.name == "cora"):
            self.graph = nx.read_edgelist("data/Cora/edges.csv", delimiter=",")

        elif (self.name == "citeseer"):
            self.graph = nx.read_edgelist("data/citeseer/citeseer.cites")

        elif (self.name == "microseer"):
            self.graph = nx.read_edgelist("data/microseer/microseer.cites")

        elif (self.name == "miniseer"):
            self.graph = nx.read_edgelist("data/miniseer/miniseer.cites")

        elif (self.name == "pubmed"):
            self.graph = nx.read_edgelist("data/Pubmed-Diabetes/data/edge_list.csv", delimiter=",")


    def train_val_test_split(self):
        np.random.seed(self.config["seed"])
        train_papers = []
        test_papers = []
        validation_papers = []
        check_breakdown = {}

        for k in self.topics:
            check_breakdown[k] = 0

        while (len(train_papers) < len(self.topics)*20):
            index = np.random.randint(len(self.index_to_paper))
            topic = self.paper_to_topic[self.index_to_paper[index]]
            if (check_breakdown[topic] < 20 and self.index_to_paper[index] not in train_papers):
                train_papers.append(self.index_to_paper[index])
                check_breakdown[topic] += 1

        while (len(validation_papers) < len(self.topics)*20):
            index = np.random.randint(len(self.index_to_paper))
            topic = self.paper_to_topic[self.index_to_paper[index]]
            if (check_breakdown[topic] < 40 and self.index_to_paper[index] not in train_papers and self.index_to_paper[index] not in validation_papers):
                validation_papers.append(self.index_to_paper[index])
                check_breakdown[topic] += 1

        for i in range(len(self.index_to_paper)):
            if (self.index_to_paper[i] not in train_papers and self.index_to_paper[i] not in validation_papers and self.index_to_paper[i] in self.paper_to_topic.keys()):
                test_papers.append(self.index_to_paper[i])

        self.train_papers = train_papers
        self.test_papers = test_papers
        self.validation_papers = validation_papers

    def train_val_test_split_small(self):
        np.random.seed(self.config["seed"])
        train_papers = []
        test_papers = []
        validation_papers = []
        check_breakdown = {}

        for k in self.topics:
            check_breakdown[k] = 0

        while (len(train_papers) < len(self.topics)*5):
            index = np.random.randint(len(self.index_to_paper))
            topic = self.paper_to_topic[self.index_to_paper[index]]
            if (check_breakdown[topic] < 10 and self.index_to_paper[index] not in train_papers):
                train_papers.append(self.index_to_paper[index])
                check_breakdown[topic] += 1

        while (len(validation_papers) < len(self.topics)*5):
            index = np.random.randint(len(self.index_to_paper))
            topic = self.paper_to_topic[self.index_to_paper[index]]
            if (check_breakdown[topic] < 20 and self.index_to_paper[index] not in train_papers and self.index_to_paper[index] not in validation_papers):
                validation_papers.append(self.index_to_paper[index])
                check_breakdown[topic] += 1

        for i in range(len(self.index_to_paper)):
            if (self.index_to_paper[i] not in train_papers and self.index_to_paper[i] not in validation_papers and self.index_to_paper[i] in self.paper_to_topic.keys()):
                test_papers.append(self.index_to_paper[i])

        self.train_papers = train_papers
        self.test_papers = test_papers
        self.validation_papers = validation_papers

def load_network(graph, config):
    model = snm.SNN()
    # Read paper to paper edge list
    topic_neurons = {}
    # Create paper neurons
    paper_neurons = {}
    feature_neurons = {}
    i = 0
    variation_scale = .01
    for node in graph.graph.nodes:
        if (node not in graph.paper_to_topic.keys()):
            continue
        if node in graph.train_papers:
            neuron = model.create_neuron(threshold=config["paper_threshold"], leak=config["paper_leak"], refractory_period=config["train_ref"])
        elif node in graph.validation_papers:
            neuron = model.create_neuron(threshold=config["paper_threshold"], leak=config["paper_leak"], refractory_period=config["validation_ref"])
        elif node in graph.test_papers:
            neuron = model.create_neuron(threshold=config["paper_threshold"], leak=config["paper_leak"], refractory_period=config["test_ref"])
        paper_neurons[node] = neuron


    for t in graph.topics:
        neuron = model.create_neuron(threshold=config["topic_threshold"], leak=config["topic_leak"], refractory_period=0)
        topic_neurons[t] = neuron

    # Create feature neurons if features are enabled
    if config["features"] == 1:
        #print("num features: ", graph.num_features)
        for i in range(graph.num_features):
            neuron_id = model.create_neuron(
                threshold=config["feature_threshold"],
                leak=config["feature_leak"],
                reset_state=0.0,
                refractory_period=config["feature_ref"],
            )
            feature_neurons[i] = neuron_id
        #print("Feature neurons: ", len(feature_neurons))

    for edge in graph.graph.edges:
        if (edge[0] not in graph.paper_to_topic.keys() or edge[1] not in graph.paper_to_topic.keys()):
            continue
        pre = paper_neurons[edge[0]]
        post = paper_neurons[edge[1]]
        graph_weight = 100.0
        variations = (1.0 + np.random.normal(0, variation_scale))
        graph_delay = 1
        if pre != post:
            model.create_synapse(pre, post, weight=config["graph_weight"], delay=config["graph_delay"], stdp_enabled=False)
            model.create_synapse(post, pre, weight=config["graph_weight"], delay=config["graph_delay"], stdp_enabled=False)

    for paper in graph.train_papers:
        paper_neuron = paper_neurons[paper]
        topic_neuron = topic_neurons[graph.paper_to_topic[paper]]
        train_to_topic_w = 1.0
        train_to_topic_w *= (1.0 + np.random.normal(0, variation_scale))
        train_to_topic_d = 1
        model.create_synapse(paper_neuron, topic_neuron, weight=config["train_to_topic_weight"], delay=config["train_to_topic_delay"], stdp_enabled=False)
        model.create_synapse(topic_neuron, paper_neuron, weight=config["train_to_topic_weight"], delay=config["train_to_topic_delay"], stdp_enabled=False)

    for paper in graph.validation_papers:
        for topic in graph.topics:
            paper_neuron = paper_neurons[paper]
            topic_neuron = topic_neurons[topic]

            validation_to_topic_w = 0.001
            validation_to_topic_w *= (1.0 + np.random.normal(0, variation_scale))
            validation_to_topic_d = 1
            model.create_synapse(paper_neuron, topic_neuron, stdp_enabled=True, weight=config["validation_to_topic_weight"], delay=config["validation_to_topic_delay"])
            model.create_synapse(topic_neuron, paper_neuron, stdp_enabled=True, weight=config["validation_to_topic_weight"], delay=config["validation_to_topic_delay"])

    for paper in graph.test_papers:
        for topic in graph.topics:
            paper_neuron = paper_neurons[paper]
            topic_neuron = topic_neurons[topic]
            test_to_topic_w = 0.001
            test_to_topic_w *= (1.0 + np.random.normal(0, variation_scale))
            test_to_topic_d = 1
            model.create_synapse(paper_neuron, topic_neuron, stdp_enabled=True, weight=config["test_to_topic_weight"], delay=config["test_to_topic_delay"])
            model.create_synapse(topic_neuron, paper_neuron, stdp_enabled=True, weight=config["test_to_topic_weight"], delay=config["test_to_topic_delay"])

    # Connect features to paper neurons if features are enabled
    if config["features"] == 1:
        for node in graph.graph.nodes:
            if (node not in graph.features.keys()):
                print("No features for node: ", node)
                continue
            paper_id = paper_neurons[node]
            for i, feature_value in enumerate(graph.features[node]):
                if feature_value == 1:
                    feature_id = feature_neurons[i]
                    variation_scale = 0.01

                    w = config["paper_to_feature_weight"]
                    w *= (1.0 + np.random.normal(0, variation_scale))
                    d = int(config["paper_to_feature_delay"])
                    stdp_on = config["paper_to_feature_stdp"]
                    # Adjust weights if weighted connections are enabled
                    if config.get("paper_to_feature_weighted", False):
                        w_paper_to_feature = (
                            config["paper_to_feature_weight"]
                            * graph.paper_to_features[node]
                            / len(graph.features[node])
                        )
                        w_feature_to_paper = (
                            config["feature_to_paper_weight"]
                            * graph.feature_to_papers[i]
                            / len(graph.features)
                        )
                    else:
                        w_paper_to_feature = w
                        w_feature_to_paper = w

                    # Create synapses
                    model.create_synapse(
                        paper_id,
                        feature_id,
                        weight=w_paper_to_feature,
                        delay=d,
                        stdp_enabled=stdp_on,
                    )
                    model.create_synapse(
                        feature_id,
                        paper_id,
                        weight=w_feature_to_paper,
                        delay=d,
                        stdp_enabled=stdp_on,
                    )

    if config.get("feature_to_topic", 0) == 1:
        # build co‑occurrence counts
        topic_feature_counts = {
            t: np.zeros(graph.num_features, dtype=int)
            for t in graph.topics
        }
        for paper_id, fv in graph.features.items():
            topic = graph.paper_to_topic[paper_id]
            topic_feature_counts[topic] += np.array(fv, dtype=int)

        for i, feat_neuron in feature_neurons.items():
            total_with_i = graph.feature_to_papers.get(i, 0)
            if total_with_i == 0:
                continue
            for topic, topic_neuron in topic_neurons.items():
                count_in_topic = int(topic_feature_counts[topic][i])
                init_ratio = count_in_topic / total_with_i

                w0 = config["feature_to_topic_weight"] * init_ratio
                d0 = config["feature_to_topic_delay"]
                stdp0 = bool(config["feature_to_topic_stdp"])

                # feature to topic
                model.create_synapse(
                    feat_neuron, topic_neuron,
                    weight=w0, delay=d0, stdp_enabled=stdp0
                )
                # topic to feature
                model.create_synapse(
                    topic_neuron, feat_neuron,
                    weight=w0, delay=d0, stdp_enabled=stdp0
                )

    return paper_neurons, topic_neurons, feature_neurons, model


def test_paper(x):
    paper =  x[0]
    graph =  x[1]
    config = x[2]
    paper_neurons, topic_neurons, feature_neurons, model = load_network(graph, config)
    paper_id = paper_neurons[paper]
    model.add_spike(0, paper_id, 100.0)
    timesteps = config["simtime"]
    model.stdp_setup(time_steps=config["stdp_timesteps"],
        Apos=config["apos"], Aneg=config["aneg"] * config["stdp_timesteps"], negative_update=True, positive_update=True)
    model.setup()
#     with open("pre_sim_model.pkl", "wb") as f:
#         pickle.dump(model, f)
    model.simulate(time_steps=timesteps)
#     with open("post_sim_model.pkl", "wb") as f:
#         pickle.dump(model, f)u

    num_spikes = np.sum(np.array(model.spike_train))
#     print(num_spikes)
    min_weight = -1000
    second_best = -1
    # Analyze the weights between the test paper neuron and topic neurons
    min_topic = None
    for topic_id, topic_paper_id in topic_neurons.items():
        # Find the synapse from topic neuron to test paper neuron
        for i, (pre, post) in enumerate(zip(model.pre_synaptic_neuron_ids, model.post_synaptic_neuron_ids)):
            if pre == paper_id and post == topic_paper_id:
                synapse_indices = i
        if synapse_indices:
            idx = synapse_indices
            weight = model.synaptic_weights[idx]
#             print(f"Topic: {topic_id}, Paper: {paper}, Weight: {weight}")
            if weight > min_weight:
                second_best = min_weight
                min_weight = weight
                min_topic = topic_id

    #if(min_weight == second_best):
    #    print("Tie!")
    #print("Best: ", min_weight)
    #print("Second: ", second_best)
    actual_topic = graph.paper_to_topic[paper]
    retval = 1 if actual_topic == min_topic else 0
#     if retval == 1:
#         print(f"MIN VAL for {paper} Topic {min_topic} CORRECT")
#     else:
#         print(f"MIN VAL for {paper} Topic {min_topic} WRONG, Expected {actual_topic}")

    return retval, num_spikes


BO_TESTING_MODE = False
PROJECT_NAME = "GNN-BO"
DEFAULT_CONFIG = None

def solve_gnn(arg1, arg2, arg3, arg4,
              arg5, arg6, arg7, arg8,
              arg9, arg10, arg11, arg12,
              arg13, arg14) -> float:
    """
    Solve the GNN model with the specific BO configuration.
    """
    local_config = copy.deepcopy(DEFAULT_CONFIG)

    override_config = DictConfig({
        "apos": [float(x) for x in [arg1, arg2, arg3, arg4, arg5]],
        "aneg": [float(arg6)],
        "mode": "validation",
        "paper_threshold": float(arg7),
        "topic_threshold": float(arg8),
        "feature_threshold": float(arg9),
        "graph_weight": float(arg10),
        "train_to_topic_weight": float(arg11),
        "test_to_topic_weight": float(arg12),
        "validation_to_topic_weight": float(arg12),
        "feature_to_topic_weight": float(arg13),
        "paper_to_feature_weight": float(arg14),
        "feature_to_paper_weight": float(arg14)

    })

    # print("Override config:")
    # print(json.dumps(OmegaConf.to_container(override_config), indent=4))

    # Extract keys from both configs
    baseline_keys = set(OmegaConf.to_container(local_config, resolve=True).keys())
    override_keys = set(OmegaConf.to_container(override_config, resolve=True).keys())

    # Identify keys that are in override_config but not in local_config
    unexpected_keys = override_keys - baseline_keys

    if unexpected_keys:
        print(f"Unexpected keys in override_config that do not exist in local_config: {unexpected_keys}")
        raise ValueError(f"Unexpected keys in override_config that do not exist in local_config: {unexpected_keys}")

    local_config = omegaconf.OmegaConf.merge(local_config, override_config)

    local_config = OmegaConf.to_container(local_config)

    np.random.seed(local_config["seed"])

     # Spike each of the test papers in simulation
    papers = []

    if BO_TESTING_MODE:
        accuracy = np.random.rand()
    else:
        graph = GraphData(local_config["dataset"], local_config)    

        with Pool(local_config["processes"]) as pool:
            if local_config["mode"] == "validation":
                iterator = graph.validation_papers
            elif local_config["mode"] == "test":
                iterator = graph.test_papers
            
            for paper in iterator:
                papers.append([paper, graph, local_config])

            # predictions will be a list of tuples: (paper, predicted_topic, retval)
            start_time = time.time()
            predictions = pool.map(test_paper, papers)
            end_time = time.time()
            print(f"Time taken {end_time - start_time}s")

            if local_config["mode"] == "validation":
                accuracy = sum(i[0] for i in predictions) / len(graph.validation_papers)
                print("Validation Accuracy:", accuracy)
            elif local_config["mode"] == "testing":
                accuracy = sum(i[0] for i in predictions) / len(graph.test_papers)
                print("Testing Accuracy:", accuracy)


    # wandb.log({"accuracy": accuracy})
    # wandb_run.finish()

    return accuracy


def main(config: omegaconf.DictConfig) -> None:
    """
    Main function to execute the script.

    Args:
        config (omegaconf.DictConfig): Merged configuration.
    """

    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config

    search_space = Space([
        Real(0.01, 100.0, name="apos_0"),
        Real(0.001, 10.0, name="apos_1"),
        Real(0.001, 1.00, name="apos_2"),
        Real(0.0001, 1.0, name="apos_3"),
        Real(0.00001,0.1, name="apos_4"),
        Real(0.000001, 0.01, name="aneg_0"),
        Real(0.00001, 10.0, name="paper_threshold"),
        Real(0.00001, 10.0, name="topic_threshold"),
        Real(0.00001, 10.0, name="feature_threshold"),
        Real(0.00001, 100.0, name="graph_weight"),
        Real(0.00001, 10.0, name="train_to_topic_weight"),
        Real(0.00001, 10.0, name="test_to_topic_weight"),
        Real(0.00001, 10.0, name="feature_to_topic_weight"),
        Real(0.00001, 10.0, name="feature_to_paper_weight")
    ])

    np.random.seed(config.seed)

    num_iterations: int = config.bo.max_iterations
    num_initial_points: int = config.bo.num_initial_points

    optimizer = Optimizer(
        dimensions=search_space,
        base_estimator="GP",
        initial_point_generator="random",
        random_state=config.seed,
    )

    parameter_log = []
    accuracy_log = []
    ask_time_log = []
    eval_time_log = []
    tell_time_log = []

    os.makedirs(config.bo.output_dir, exist_ok=False)
    param_log_path: str = os.path.join(config.bo.output_dir, "parameter_log.npy")
    accuracy_log_path: str = os.path.join(config.bo.output_dir, "accuracy_log.npy")
    ask_time_log_path: str = os.path.join(config.bo.output_dir, "ask_time_log.npy")
    eval_time_log_path: str = os.path.join(config.bo.output_dir, "eval_time_log.npy")
    tell_time_log_path: str = os.path.join(config.bo.output_dir, "tell_time_log.npy")
    result_path: str = os.path.join(config.bo.output_dir, "result.pkl")


    for idx in range(num_initial_points + num_iterations):
        if idx <= num_initial_points:
            print(f"Initial point {idx + 1}/{num_initial_points}")
        else:
            print(f"Optimization iteration {idx + 1 - num_initial_points}/{num_iterations}")


        ask_time_start: float = time.time()
        next_x = optimizer.ask()
        ask_time_end: float = time.time()

        eval_time_start: float = time.time()
        f_val = solve_gnn(*next_x)
        eval_time_end: float = time.time()

        parameter_log.append(next_x[0])
        accuracy_log.append(f_val)

        tell_time_start: float = time.time()
        optimizer_result = optimizer.tell(next_x, -1 * f_val)
        tell_time_end: float = time.time()

        ask_time_log.append(ask_time_end - ask_time_start)
        eval_time_log.append(eval_time_end - eval_time_start)
        tell_time_log.append(tell_time_end - tell_time_start)


        # save the logs very iteration
        if idx % config.bo.save_iteration == 0:

            np.save(param_log_path, np.array(parameter_log))
            np.save(accuracy_log_path, np.array(accuracy_log))
            np.save(ask_time_log_path, np.array(ask_time_log))
            np.save(eval_time_log_path, np.array(eval_time_log))
            np.save(tell_time_log_path, np.array(tell_time_log))

            with open(result_path, "wb") as f:
                pickle.dump(optimizer_result, f)

            # plot the accuracy log sorted and unsorted
            plt.plot(accuracy_log)
            plt.title("Accuracy vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.savefig(os.path.join(config.bo.output_dir, "accuracy_plot.png"))
            plt.close()

            # plot the accuracy log sorted
            sorted_accuracy_log = np.array(accuracy_log)
            sorted_accuracy_log.sort()
            plt.plot(sorted_accuracy_log)
            plt.title("Sorted Accuracy vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.savefig(os.path.join(config.bo.output_dir, "sorted_accuracy_plot.png"))
            plt.close()

            # plot the ask time log
            plt.plot(ask_time_log)
            plt.title("Ask Time vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Time (s)")
            plt.savefig(os.path.join(config.bo.output_dir, "ask_time_plot.png"))
            plt.close()

            # plot the eval time log
            plt.plot(eval_time_log)
            plt.title("Eval Time vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Time (s)")
            plt.savefig(os.path.join(config.bo.output_dir, "eval_time_plot.png"))
            plt.close()

            # plot the tell time log
            plt.plot(tell_time_log)
            plt.title("Tell Time vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Time (s)")
            plt.savefig(os.path.join(config.bo.output_dir, "tell_time_plot.png"))
            plt.close()

if __name__ == "__main__":
    args, unknown_args = parse_args()

    # load global parameters from argument
    PROJECT_NAME = args.project_name
    BO_TESTING_MODE = args.test_bo

    config = omegaconf.OmegaConf.load(args.config)
    override_config = omegaconf.OmegaConf.from_dotlist(unknown_args)
    validate_overrides(config, override_config)
    print_introduction(args, config, override_config)
    config = omegaconf.OmegaConf.merge(config, override_config)
    main(config)
