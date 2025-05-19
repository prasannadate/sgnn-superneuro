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
import random
from collections import Counter
#import cProfile, pstats, csv
from collections import defaultdict
#import io

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
                #print("No features for node: ", node)
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

def tie_break_by_feature_propagation(
    paper,
    tied_topics,
    graph,
    paper_neurons,
    feature_neurons,
    topic_neurons,
    model
):
    # collect which features the paper has
    features_idx = [i for i, val in enumerate(graph.features[paper]) if val]

    def get_weight(pre_id, post_id):
        for pre, post, w in zip(
            model.pre_synaptic_neuron_ids,
            model.post_synaptic_neuron_ids,
            model.synaptic_weights
        ):
            if pre == pre_id and post == post_id:
                return w
        return 0.0

    # score each tied topic
    scores = {}
    p_nid = paper_neurons[paper]
    for topic in tied_topics:
        t_nid = topic_neurons[topic]
        total = 0.0
        for i in features_idx:
            f_nid = feature_neurons[i]

            # paper -> feature weight
            w_pf = get_weight(p_nid, f_nid)
            # feature -> topic weight
            w_ft = get_weight(f_nid, t_nid)

            total += w_pf * w_ft

        scores[topic] = total

    max_score = max(scores.values())
    winners = [t for t,s in scores.items() if s == max_score]
    return random.choice(winners)

def test_paper(x):
    paper =  x[0]
    graph =  x[1]
    config = x[2]
    paper_neurons, topic_neurons, feature_neurons, model = load_network(graph, config)
    #print(topic_neurons)
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
    #if model.spike_train == []:
    #    print(num_spikes)
#     print(num_spikes)
    weights = {}
    for topic_id, topic_paper_id in topic_neurons.items():
        w = None
        for pre, post, weight in zip(
            model.pre_synaptic_neuron_ids,
            model.post_synaptic_neuron_ids,
            model.synaptic_weights
        ):
            if pre == paper_id and post == topic_paper_id:
                w = weight
                break
        if w is not None:
            weights[topic_id] = w

    max_w = max(weights.values())

    tied_topics = [t for t, w in weights.items() if w == max_w]
    tied_topics = [t for t,w in weights.items() if w == max_w]

    # if tie, break by feature propagation
    if len(tied_topics) > 1:
        chosen = tie_break_by_feature_propagation(
            paper, tied_topics, graph,
            paper_neurons, feature_neurons, topic_neurons, model
        )
    else:
        chosen = tied_topics[0]

    retval = 1 if graph.paper_to_topic[paper] == chosen else 0
    

    return retval, num_spikes


if __name__ == '__main__':
#    pr = cProfile.Profile()
#    pr.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/cora/apos_cora_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    np.random.seed(config["seed"])
    graph = GraphData(config["dataset"], config)

    i = 0
    correct = 0
    total = 0
    num_spikes = 0

    pool = Pool(8)
    if config["mode"] == "validation":
        papers = []
        for paper in graph.validation_papers:
            papers.append([paper, graph, config])
        x = pool.map(test_paper, papers)

        valid_acc = sum(i[0] for i in x) / len(graph.validation_papers)
        num_spikes = sum(i[1] for i in x)
        print("Number of spikes:", num_spikes)
        print("Validation Accuracy:", valid_acc)
        if config["dump_json"] == 1:
            with open('results.json', 'a') as f:
                accuracy = {
                    "validation_accuracy": valid_acc
                }
                dump_data = config | accuracy
                json.dump(dump_data, f, indent=2)

    if config["mode"] == "test":
        papers = []
        for paper in graph.test_papers:
            papers.append([paper, graph, config])
        x = pool.map(test_paper, papers)
        test_acc = sum(i[0] for i in x) / len(graph.test_papers)
        num_spikes = sum(i[1] for i in x)
        print("Number of spikes:", num_spikes)
        print("Test Accuracy:", test_acc)
        if config["dump_json"] == 1:
            with open('results.json', 'a') as f:
                accuracy = {
                    "test_accuracy": test_acc
                }
                dump_data = config | accuracy
                json.dump(dump_data, f, indent=2)

#pr.disable()

# Aggregate timing by function name
#ps = pstats.Stats(pr)
#ps.sort_stats('cumulative')

#func_times = defaultdict(float)
#for func, stats in ps.stats.items():
#    _, _, func_name = func
#    cumtime = stats[3]
#    func_times[func_name] += cumtime

# Save to CSV
#with open("cprofile_summary.csv", "w", newline="") as f:
#    writer = csv.writer(f)
#    writer.writerow(["Function", "Cumulative Time (s)"])
#    for name, total in sorted(func_times.items(), key=lambda x: -x[1]):
#        writer.writerow([name, f"{total:.6f}"])
