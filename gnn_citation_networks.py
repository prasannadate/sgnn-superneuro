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
    i = 0
    variation_scale = .01
    for node in graph.graph.nodes:
        if (node not in graph.paper_to_topic.keys()):
            continue
        if node in graph.train_papers:
            neuron = model.create_neuron(threshold=config["paper_threshold"], leak=config["paper_leak"], refractory_period=config["train_ref"]).idx
        elif node in graph.validation_papers:
            neuron = model.create_neuron(threshold=config["paper_threshold"], leak=config["paper_leak"], refractory_period=config["validation_ref"]).idx
        elif node in graph.test_papers:
            neuron = model.create_neuron(threshold=config["paper_threshold"], leak=config["paper_leak"], refractory_period=config["test_ref"]).idx
        paper_neurons[node] = neuron


    for t in graph.topics:
        neuron = model.create_neuron(threshold=config["topic_threshold"], leak=config["topic_leak"], refractory_period=0).idx
        topic_neurons[t] = neuron

    for edge in graph.graph.edges:
        if (edge[0] not in graph.paper_to_topic.keys() or edge[1] not in graph.paper_to_topic.keys()):
            continue
        pre = paper_neurons[edge[0]]
        post = paper_neurons[edge[1]]
        graph_weight = 100.0
        variations = (1.0 + np.random.normal(0, variation_scale))
        graph_delay = 1
        model.create_synapse(pre, post, weight=config["graph_weight"], delay=config["graph_delay"], stdp_enabled=False)
        model.create_synapse(post, pre, weight=config["graph_weight"], delay=config["graph_delay"], stdp_enabled=False, exist="dontadd")

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


def test_paper(x):
    paper =  x[0]
    graph =  x[1]
    config = x[2]
    paper_neurons, topic_neurons, model = load_network(graph, config)
    print(model)
    paper_id = paper_neurons[paper]
    model.add_spike(0, paper_id, 100.0)
    timesteps = config["simtime"]
    model.stdp_setup(Apos=config["apos"], Aneg=config["aneg"] * config["stdp_timesteps"], negative_update=True, positive_update=True)
    # model.setup()
#     with open("pre_sim_model.pkl", "wb") as f:
#         pickle.dump(model, f)
    model.simulate(time_steps=timesteps)
#     with open("post_sim_model.pkl", "wb") as f:
#         pickle.dump(model, f)u

    num_spikes = np.sum(np.array(model.spike_train))
#     print(num_spikes)
    min_weight = -1000

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
                min_weight = weight
                min_topic = topic_id


    actual_topic = graph.paper_to_topic[paper]
    retval = 1 if actual_topic == min_topic else 0
#     if retval == 1:
#         print(f"MIN VAL for {paper} Topic {min_topic} CORRECT")
#     else:
#         print(f"MIN VAL for {paper} Topic {min_topic} WRONG, Expected {actual_topic}")

    return retval, num_spikes


if __name__ == '__main__':
    config = yaml.safe_load(open("configs/microseer/default_microseer_config.yaml"))

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
