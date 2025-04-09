import os
import time
import json
import yaml
import argparse
import tempfile
import pickle as pkl
import pathlib as pl
from multiprocessing import Pool
from dataclasses import dataclass, field

import tqdm
import numpy as np
import networkx as nx
import superneuromat as snm
# import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

wd = pl.Path(__file__).parent.absolute()  # get the Path() of this python file


class GraphData():
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.paper_to_topic = {}  # maps the paper ID in the dataset to its topic ID
        self.index_to_paper = []    # creates an index for each paper
        self.topics = []            # the list of topics
        self.train_papers = []
        self.validation_papers = []
        self.test_papers = []
        self.load_topics()
        if self.name == "microseer":
            self.train_val_test_split_small()
        else:
            self.train_val_test_split()
        self.load_features()
        self.load_graph()

    @classmethod
    def read_directed_cites_tab(cls, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return cls.read_directed_cites_lines(lines)

    @classmethod
    def read_directed_cites_lines(cls, lines):
        for line in lines:
            fields = line.strip().split()
            if len(fields) < 4:  # skip the first two lines
                continue
            _cite_id, paper, _sep, cited, *_ = fields  # extract paper id and cited paper id fields
            paper = int(paper.strip().removeprefix("paper:"))
            cited = int(cited.strip().removeprefix("paper:"))
            yield paper, cited

    def load_topics(self):
        if (self.name == "cora"):
            f = open(wd / "data/Cora/group-edges.csv", 'r')
            lines = f.readlines()
            f.close()

            for line in lines:
                fields = line.strip().split(",")
                if (fields[1] not in self.topics):
                    self.topics.append(fields[1])
                self.paper_to_topic[fields[0]] = fields[1]
                self.index_to_paper.append(fields[0])
        elif (self.name == "citeseer"):
            f = open(wd / "data/citeseer/citeseer.content", 'r')
            lines = f.readlines()
            f.close()

            for line in lines:
                fields = line.strip().split()
                if (fields[-1] not in self.topics):
                    self.topics.append(fields[-1])
                # print(fields[0])
                self.paper_to_topic[fields[0]] = fields[-1]
                self.index_to_paper.append(fields[0])
        elif (self.name == "microseer"):
            f = open(wd / "data/microseer/microseer.content", 'r')
            lines = f.readlines()
            f.close()

            for line in lines:
                fields = line.strip().split()
                if (fields[-1] not in self.topics):
                    self.topics.append(fields[-1])
                # print(fields[0])
                self.paper_to_topic[fields[0]] = fields[-1]
                self.index_to_paper.append(fields[0])
        elif (self.name == "miniseer"):
            f = open(wd / "data/miniseer/miniseer.content", 'r')
            lines = f.readlines()
            f.close()

            for line in lines:
                fields = line.strip().split()
                if (fields[-1] not in self.topics):
                    self.topics.append(fields[-1])
                # print(fields[0])
                self.paper_to_topic[fields[0]] = fields[-1]
                self.index_to_paper.append(fields[0])
        elif (self.name == "pubmed"):
            with open(wd / "data/Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab", 'r') as f:
                lines = f.readlines()  # read the file
            lines = lines[2:]  # skip the first two lines

            for line in lines:
                fields = line.strip().split()  # split on tab separator
                paper_idx, topic, *_features = fields  # extract paper name and topic/label
                if (topic not in self.topics):
                    self.topics.append(topic)
                paper_idx = int(paper_idx.strip().removeprefix("paper:"))  # make paper ID an int
                self.paper_to_topic[paper_idx] = topic  # associate paper ID (as int) with topic/label
                self.index_to_paper.append(paper_idx)

    def load_features(self):
        self.features = {}  # keyed on paper ID, value is the feature vector
        if (self.name == "cora"):
            f = open(wd / "data/Cora/cora/cora.content", 'r')
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
            f = open(wd / "data/citeseer/citeseer.content", 'r')
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
            f = open(wd / "data/microseer/microseer.content", 'r')
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
            f = open(wd / "data/miniseer/miniseer.content", 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                fields = line.strip().split()
                paper_id = fields[0]
                feature = [int(x) for x in fields[1:-1]]
                self.features[paper_id] = feature
                self.num_features = len(feature)
            print(wd / "NUM PAPERS WITH FEATURES: ", len(self.features.keys()))
        elif (self.name == "pubmed"):
            f = open(wd / "data/Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab", 'r')
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
                feature = [0] * self.num_features
                fields = line.split()
                paper_id = int(fields[0])
                xfields = fields[-1].split('=')
                xfields = xfields[1].split(',')
                for x in xfields:
                    feature[all_features[x]] = 1
                self.features[paper_id] = feature

        self.paper_to_features = {}  # keyed on paper ID, value is the number of features it has
        self.feature_to_papers = {}  # keyed on feature ID, value is the number of papers that have that feature
        for p in self.features.keys():
            self.paper_to_features[p] = np.sum(self.features[p])
            for i in range(len(self.features[p])):
                if (i not in self.feature_to_papers.keys()):
                    self.feature_to_papers[i] = 0
                self.feature_to_papers[i] += self.features[p][i]

    def load_graph(self):
        if (self.name == "cora"):
            self.graph = nx.read_edgelist(wd / "data/Cora/edges.csv", delimiter=",")

        elif (self.name == "citeseer"):
            self.graph = nx.read_edgelist(wd / "data/citeseer/citeseer.cites")

        elif (self.name == "microseer"):
            self.graph = nx.read_edgelist(wd / "data/microseer/microseer.cites")

        elif (self.name == "miniseer"):
            self.graph = nx.read_edgelist(wd / "data/miniseer/miniseer.cites")

        elif (self.name == "pubmed"):
            # self.graph = nx.read_edgelist(wd / "data/Pubmed-Diabetes/data/edge_list.csv", delimiter=",")
            self.graph = nx.from_edgelist(self.read_directed_cites_tab(  # above broken for me so I wrote new one -kz-apr'25
                wd / "data/Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab"
            ))

    def train_val_test_split(self):
        np.random.seed(self.config["seed"])
        train_papers = []
        test_papers = []
        validation_papers = []
        check_breakdown = {}

        for k in self.topics:
            check_breakdown[k] = 0

        while (len(train_papers) < len(self.topics) * 20):
            index = np.random.randint(len(self.index_to_paper))
            topic = self.paper_to_topic[self.index_to_paper[index]]
            if (check_breakdown[topic] < 20 and self.index_to_paper[index] not in train_papers):
                train_papers.append(self.index_to_paper[index])
                check_breakdown[topic] += 1

        while (len(validation_papers) < len(self.topics) * 20):
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

        while (len(train_papers) < len(self.topics) * 5):
            index = np.random.randint(len(self.index_to_paper))
            topic = self.paper_to_topic[self.index_to_paper[index]]
            if (check_breakdown[topic] < 10 and self.index_to_paper[index] not in train_papers):
                train_papers.append(self.index_to_paper[index])
                check_breakdown[topic] += 1

        while (len(validation_papers) < len(self.topics) * 5):
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
    model = snm.NeuromorphicModel()
    # Read paper to paper edge list
    topic_neurons = {}
    # Create paper neurons
    paper_neurons = {}
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
        # paper_neurons[node] = neuron
        paper_neurons[node] = neuron.idx

    for t in graph.topics:
        neuron = model.create_neuron(threshold=config["topic_threshold"], leak=config["topic_leak"], refractory_period=0)
        # topic_neurons[t] = neuron
        topic_neurons[t] = neuron.idx

    for edge in graph.graph.edges:
        if (edge[0] not in graph.paper_to_topic.keys() or edge[1] not in graph.paper_to_topic.keys()):
            continue
        pre = paper_neurons[edge[0]]
        post = paper_neurons[edge[1]]
        graph_weight = 100.0
        variations = (1.0 + np.random.normal(0, variation_scale))
        graph_delay = 1
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

    return paper_neurons, topic_neurons, model


def create_model(graph, config):
    paper_neurons, topic_neurons, model = load_network(graph, config)
    model.stdp_setup(time_steps=config["stdp_timesteps"],
        Apos=config["apos"], Aneg=config["aneg"] * config["stdp_timesteps"], negative_update=True, positive_update=True)
    return model, paper_neurons, topic_neurons


def test_paper_from_pickle(x):
    paper_id, temp = x
    with open(temp, 'rb') as f:
        d = pkl.load(f)
    return test_paper((paper_id, d))


def test_paper(x):
    paper_id, d = x
    model = d["model"]
    graph = d["graph"]
    paper_neurons = d["paper_neurons"]
    topic_neurons = d["topic_neurons"]
    config = d["config"]
    model.add_spike(0, paper_neurons[paper_id], 100.0)
    # model.setup()
    model.simulate(time_steps=config["simtime"], use='gpu')

    def get_synapse(model, pre_id, post_id):
        for i, (pre, post) in enumerate(zip(model.pre_synaptic_neuron_ids, model.post_synaptic_neuron_ids)):
            if pre_id == pre and post_id == post:
                return i

    # Analyze the weights between the test paper neuron and topic neurons
    topic_weights = {}
    for topic_id, topic_paper_id in topic_neurons.items():
        # Find the synapse from topic neuron to test paper neuron
        synapse_idx = get_synapse(model, paper_neurons[paper_id], topic_paper_id)
        if synapse_idx:
            weight = model.synaptic_weights[synapse_idx]
            topic_weights[topic_id] = weight
            # print(f"Topic: {topic_id}, Paper: {paper_id}, Weight: {weight}")

    best_topic = max(topic_weights, key=topic_weights.get)

    actual_topic = graph.paper_to_topic[paper_id]
    retval = actual_topic == best_topic
    # if retval:
    #     print(f"MIN VAL for {paper} Topic {min_topic} CORRECT")
    # else:
    #     print(f"MIN VAL for {paper} Topic {min_topic} WRONG, Expected {actual_topic}")
    # del graph, model, paper_neurons, topic_neurons, config, d, temp, x  # unload to save memory
    return retval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default=wd / 'configs/miniseer/default_miniseer_config.yaml')
    parser.add_argument('--config', type=str, default=wd / 'configs/pubmed/default_pubmed_config.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    np.random.seed(config["seed"])
    graph = GraphData(config["dataset"], config)
    model, paper_neurons, topic_neurons = create_model(graph, config)

    # save model to file for multiprocessing
    d = {
        'graph': graph,
        'config': config,
        'model': model,
        'paper_neurons': paper_neurons,
        'topic_neurons': topic_neurons,
    }
    temp = tempfile.NamedTemporaryFile(delete=False)
    with open(temp.name, 'wb') as f:
        pkl.dump(d, f)
    papers = []

    mode = config['mode']
    if mode == 'test':
        papers = graph.test_papers
    if mode == "validation":
        papers = graph.validation_papers
    n = len(papers)
    bundles = [(paperstr, temp.name) for paperstr in papers]
    del papers, graph, model, paper_neurons, topic_neurons  # unload to save memory

    start = time.time()

    # single-process evaluation
    # bundles = [(paperstr, d) for paperstr in papers]
    # x = [test_paper(bundle) for bundle in tqdm.tqdm(bundles)]

    try:  # multi-process evaluation
        x = process_map(test_paper, bundles, max_workers=2)  # with tqdm
        # pool = Pool(2)
        # x = pool.map(test_paper, bundles)
        pass
    finally:  # clean up and delete the temp file no matter what
        temp.close()  # close the temp file
        os.unlink(temp.name)  # DON'T COMMENT OUT THIS.
    end = time.time()
    print(f"Time taken: {end - start}")

    accuracy = np.sum(x) / n
    print(f"{mode.title()} Accuracy:", accuracy)
    if config["dump_json"] == 1:
        with open('results.json', 'a') as f:
            accuracy = {
                f"{mode}_accuracy": accuracy
            }
            dump_data = config | accuracy
            json.dump(dump_data, f, indent=2)
