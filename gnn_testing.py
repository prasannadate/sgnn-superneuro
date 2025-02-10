#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from superneuromat.neuromorphicmodel import NeuromorphicModel
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from multiprocessing import Pool




# In[2]:


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
        self.train_val_test_split()
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


# In[3]:


def load_network(graph, config):
    model = NeuromorphicModel()
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
        paper_neurons[node] = neuron


    for t in graph.topics:
        neuron = model.create_neuron(threshold=config["topic_threshold"], leak=config["topic_leak"], refractory_period=0)
        topic_neurons[t] = neuron

    for edge in graph.graph.edges:
        if (edge[0] not in graph.paper_to_topic.keys() or edge[1] not in graph.paper_to_topic.keys()):
            continue
        pre = paper_neurons[edge[0]]
        post = paper_neurons[edge[1]]
        graph_weight = 100.0
        variations = (1.0 + np.random.normal(0, variation_scale))
        graph_delay = 1
        model.create_synapse(pre, post, weight=config["graph_weight"], delay=config["graph_delay"], enable_stdp=False)
        model.create_synapse(post, pre, weight=config["graph_weight"], delay=config["graph_delay"], enable_stdp=False)

    for paper in graph.train_papers:
        paper_neuron = paper_neurons[paper]
        topic_neuron = topic_neurons[graph.paper_to_topic[paper]]
        train_to_topic_w = 1.0
        train_to_topic_w *= (1.0 + np.random.normal(0, variation_scale))
        train_to_topic_d = 1
        model.create_synapse(paper_neuron, topic_neuron, weight=config["train_to_topic_weight"], delay=config["train_to_topic_delay"], enable_stdp=False)
        model.create_synapse(topic_neuron, paper_neuron, weight=config["train_to_topic_weight"], delay=config["train_to_topic_delay"], enable_stdp=False)

    for paper in graph.validation_papers:
        for topic in graph.topics:
            paper_neuron = paper_neurons[paper]
            topic_neuron = topic_neurons[topic]

            validation_to_topic_w = 0.001
            validation_to_topic_w *= (1.0 + np.random.normal(0, variation_scale))
            validation_to_topic_d = 1
            model.create_synapse(paper_neuron, topic_neuron, enable_stdp=True, weight=config["validation_to_topic_weight"], delay=config["validation_to_topic_delay"])
            model.create_synapse(topic_neuron, paper_neuron, enable_stdp=True, weight=config["validation_to_topic_weight"], delay=config["validation_to_topic_delay"])

    for paper in graph.test_papers:
        for topic in graph.topics:
            paper_neuron = paper_neurons[paper]
            topic_neuron = topic_neurons[topic]
            test_to_topic_w = 0.001
            test_to_topic_w *= (1.0 + np.random.normal(0, variation_scale))
            test_to_topic_d = 1
            model.create_synapse(paper_neuron, topic_neuron, enable_stdp=True, weight=config["test_to_topic_weight"], delay=config["test_to_topic_delay"])
            model.create_synapse(topic_neuron, paper_neuron, enable_stdp=True, weight=config["test_to_topic_weight"], delay=config["test_to_topic_delay"])




    return paper_neurons, topic_neurons, model




# In[4]:


def test_paper(x):
    paper =  x[0]
    graph =  x[1]
    config = x[2]
    paper_neurons, topic_neurons, model = load_network(graph, config)
    paper_id = paper_neurons[paper]
    model.add_spike(0, paper_id, 100.0)
    timesteps = config["simtime"]
    model.stdp_setup(time_steps=config["stdp_timesteps"],
        Apos=config["apos"], Aneg=config["aneg"] * config["stdp_timesteps"], negative_update=True, positive_update=True)
    model.setup()
    model.simulate(time_steps=timesteps)
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
            print(f"Topic: {topic_id}, Paper: {paper}, Weight: {weight}")
            if weight > min_weight:
                min_weight = weight
                min_topic = topic_id


    actual_topic = graph.paper_to_topic[paper]
    retval = 1 if actual_topic == min_topic else 0
    if retval == 1:
        print(f"MIN VAL for {paper} Topic {min_topic} CORRECT")
    else:
        print(f"MIN VAL for {paper} Topic {min_topic} WRONG, Expected {actual_topic}")

    return retval

                #need to setup a list of zipped synapses so I can find the ids and get the weight
                #check above where I setup synapses to get the lists to iterate over




# In[ ]:

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GNN-SNN")
    parser.add_argument("--dataset", "-d", type=str, choices=["cora", "citeseer", "pubmed"], required=True)
    parser.add_argument("--mode", "-m", type=str, choices=["validation", "test"], required=True)
    parser.add_argument("--seed", "-s", type=int, default=0)
#     parser.add_argument("--features", type=int, choices=[0,1])
    parser.add_argument("--paper_leak", type=float, default=0.0)           # [1., 10., 100., 100000.]
    parser.add_argument("--paper_threshold", type=float, default=1.0)            # [0.25, 0.5, 0.75, 1.0, 1.5]
    parser.add_argument("--train_ref", type=int, default=1)                 # [1.0, 5.0, 10.0, 100.0, 1000.0]
#     parser.add_argument("--feature_ref", type=float, default=1.0)                 # [1.0, 5.0, 10.0, 100.0, 1000.0]
    parser.add_argument("--validation_ref", type=int, default=1000)         # [1.0, 5.0, 10.0, 100.0, 1000.0]
    parser.add_argument("--apos", type=list, default=[1.0, .5, .1, .01, .001])
    parser.add_argument("--aneg", type=list, default=[.0001])
    parser.add_argument("--stdp_timesteps", type=int, default=5)

    # NOTE: We probably want to keep this parameter the same as the validation parameter above
    parser.add_argument("--test_ref", type=int, default=1000)               # [1.0, 5.0, 10.0, 100.0, 1000.0]
    parser.add_argument("--topic_leak", type=float, default=0.0)           # [1., 10., 100., 100000.]
    parser.add_argument("--topic_threshold", type=float, default=1.)            # [0.25, 0.5, 0.75, 1.0, 1.5]
#     parser.add_argument("--feature_leak", type=float, default=100000.0)         # [1., 10., 100., 100000.]
#     parser.add_argument("--feature_threshold", type=float, default=1.0)         # [0.25, 0.5, 0.75, 1.0, 1.5]
#     parser.add_argument("--feature_tau_minus", type=float, default=30.0)        # [5., 10., 15., 20., 25., 30., 40., 50.]
    parser.add_argument("--graph_weight", type=float, default=100.0)            # [0.5, 1.0, 10.0, 100.0]
    parser.add_argument("--graph_delay", type=int, default=1)               # [1., 2., 5., 10., 20.]
    parser.add_argument("--train_to_topic_weight", type=float, default=1.0)     # [0.5, 1.0, 10.0, 100.0]
    parser.add_argument("--train_to_topic_delay", type=int, default=1)      # [1., 2., 5., 10., 20.]
    parser.add_argument("--validation_to_topic_weight", type=float, default=0.001)  # [0.001, 0.005, 0.01, 0.05, 0.1]
    parser.add_argument("--validation_to_topic_delay", type=int, default=1)     # [1., 2., 5., 10., 20.]

    # NOTE: We probably want to keep these parameters the same as the validation set parameters
    parser.add_argument("--test_to_topic_weight", type=float, default=0.001)    # [0.001, 0.005, 0.01, 0.05, 0.1]
    parser.add_argument("--test_to_topic_delay", type=int, default=1)       # [1., 2., 5., 10., 20.]
#     parser.add_argument("--paper_to_feature_weight", type=float, default=0.2)   # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     parser.add_argument("--feature_to_paper_weight", type=float, default=0.2)   # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     parser.add_argument("--paper_to_feature_delay", type=float, default=4.0)    # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    parser.add_argument("--simtime", type=int, default=10)                  # [5., 10., 20., 30., 40., 50. 75., 100., 150., 200.]
#     parser.add_argument("--processes", type=int, default=4)         # Change this depending on your machine
#     parser.add_argument("--monitors", type=str, default="False")
#     parser.add_argument("--paper_to_feature_stdp", type=str, default="False")
#     parser.add_argument("--paper_to_feature_weighted", type=str, default="False")


    args = parser.parse_args()

    config = vars(args)

    np.random.seed(args.seed)
    graph = GraphData(args.dataset, config)

    i = 0
    correct = 0
    total = 0

    pool = Pool(4)
    if args.mode == "validation":
        papers = []
        for paper in graph.validation_papers:
            papers.append([paper, graph, config])
        x = pool.map(test_paper, papers)
        valid_acc = np.sum(x) / len(graph.validation_papers)
        print("Validation Accuracy:", np.sum(x) / len (graph.validation_papers))
        with open('commandline_args.json', 'a') as f:
            accuracy = {
                "validation_accuracy": valid_acc
            }
            dump_data = args.__dict__ | accuracy
            json.dump(dump_data, f, indent=2)

    if args.mode == "test":
        papers = []
        for paper in graph.test_papers:
            papers.append([paper, graph, config])
        x = pool.map(test_paper, papers)
        test_acc = np.sum(x) / len(graph.test_papers)
        print("Test Accuracy:", np.sum(x) / len (graph.test_papers))
        with open('commandline_args.json', 'a') as f:
            accuracy = {
                "test_accuracy": test_acc
            }
            dump_data = args.__dict__ | accuracy
            json.dump(dump_data, f, indent=2)


# In[ ]:





# In[ ]:


#weights beyond 10.0 for the papers and topics cause a decrease
#5.0 is the best
#so far changing the graph weights and apos, aneg does not affect the results. The validation weight does.
#graph weight 25.0 -> 50.0 (the current best % is 34)
#with raising the weight of the validation synapse i got to 35.8%
