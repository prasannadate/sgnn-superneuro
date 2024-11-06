# NOTE: Code taken as it is from: https://code.ornl.gov/schumancd/neuromorphic-gnn/-/tree/main?ref_type=heads

import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse
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
                print(fields[0])
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

def load_network(graph, config):
    import nest
    nest.set_verbosity("M_QUIET")
    nest.ResetKernel()
    topic_neurons = {}
    paper_neurons = {}
    paper_voltmeters = {}
    paper_recorders = {}
    topic_recorders = {}
    for node in graph.graph.nodes:
        if (node not in graph.paper_to_topic.keys()):
            continue
        neuron = nest.Create("iaf_psc_delta")
        paper_neurons[node] = neuron
        nest.SetStatus(neuron, {"tau_m":config["paper_leak"]}) # Turn off leak
        nest.SetStatus(neuron, {"V_th":config["paper_threshold"]}) # We're going to need to mess with this
        nest.SetStatus(neuron, {"V_m":0.})
        nest.SetStatus(neuron, {"V_reset":0.})
        nest.SetStatus(neuron, {"E_L":0})
        nest.SetStatus(neuron, {"tau_minus":config["paper_tau_minus"]}) # We're going to need to mess with this 
        if (node in graph.train_papers):
            nest.SetStatus(neuron, {"t_ref":config["train_ref"]})
        elif (node in graph.validation_papers):
            nest.SetStatus(neuron, {"t_ref":config["validation_ref"]})
        elif (node in graph.test_papers):
            nest.SetStatus(neuron, {"t_ref":config["test_ref"]})

        if (config["monitors"] == True):
            voltmeter = nest.Create("voltmeter")
            nest.Connect(voltmeter, neuron)
            paper_voltmeters[node] = voltmeter        
            recorder = nest.Create('spike_recorder')
            nest.Connect(neuron, recorder)
            paper_recorders[node] = recorder 

    topic_voltmeters = {}

    # Create topic neurons
    for t in graph.topics:
        neuron = nest.Create("iaf_psc_delta")
        topic_neurons[t] = neuron
        nest.SetStatus(neuron, {"tau_m":config["topic_leak"]}) # Turn off leak
        nest.SetStatus(neuron, {"V_th":config["topic_threshold"]}) # We're going to need to mess with this
        nest.SetStatus(neuron, {"V_m":0})
        nest.SetStatus(neuron, {"V_reset":0})
        nest.SetStatus(neuron, {"E_L":0})
        nest.SetStatus(neuron, {"tau_minus":config["topic_tau_minus"]}) # We're going to need to mess with this
  
        if (config["monitors"] == True):
            voltmeter = nest.Create("voltmeter")
            nest.Connect(voltmeter, neuron)
            topic_voltmeters[t] = voltmeter 
            recorder = nest.Create('spike_recorder')
            nest.Connect(neuron, recorder)
            topic_recorders[t] = recorder


    feature_neurons = {}
    feature_voltmeters = {}
    # Create features
    if (config["features"] == 1):
        for i in range(graph.num_features):
            neuron = nest.Create("iaf_psc_delta")
            feature_neurons[i] = neuron
            nest.SetStatus(neuron, {"tau_m":config["feature_leak"]})
            nest.SetStatus(neuron, {"V_th":config["feature_threshold"]})
            nest.SetStatus(neuron, {"V_m":0})
            nest.SetStatus(neuron, {"V_reset":0})
            nest.SetStatus(neuron, {"tau_minus":config["feature_tau_minus"]})
            nest.SetStatus(neuron, {"t_ref":config["feature_ref"]})
            nest.SetStatus(neuron, {"E_L":0})

            if (config["monitors"] == True):
                voltmeter = nest.Create("voltmeter")
                nest.Connect(voltmeter, neuron)
                feature_voltmeters[i] = voltmeter
 
    for edge in graph.graph.edges:
        if (edge[0] not in graph.paper_to_topic.keys() or edge[1] not in graph.paper_to_topic.keys()):
            continue
        pre = paper_neurons[edge[0]]
        post = paper_neurons[edge[1]]
        w = config["graph_weight"]  
        d = config["graph_delay"]
        nest.Connect(pre, post, syn_spec={"weight" : w, "delay" : d})
        nest.Connect(post, pre, syn_spec={"weight": w, "delay": d})


    for paper in graph.train_papers:
        paper_neuron = paper_neurons[paper]
        topic_neuron = topic_neurons[graph.paper_to_topic[paper]]
        w = config["train_to_topic_weight"]
        d = config["train_to_topic_delay"]
        nest.Connect(paper_neuron, topic_neuron, syn_spec={"weight": w, "delay": d})
        nest.Connect(topic_neuron, paper_neuron, syn_spec={"weight": w, "delay": d})

    wr = nest.Create('weight_recorder')
    nest.CopyModel("stdp_synapse", "stdp_synapse_rec", {"weight_recorder": wr})

    for paper in graph.validation_papers:
        for topic in graph.topics:
            paper_neuron = paper_neurons[paper]
            topic_neuron = topic_neurons[topic]
            w = config["validation_to_topic_weight"]
            d = config["validation_to_topic_delay"]
            alpha = config["validation_to_topic_alpha"]
            tau_plus = config["validation_to_topic_tau_plus"]
            nest.Connect(paper_neuron, topic_neuron, syn_spec={"synapse_model": "stdp_synapse", "alpha": alpha, "weight" : w, "delay" : d, "tau_plus" : tau_plus}) # Need to figure out how to do STDP on this synapse with NEST
            nest.Connect(topic_neuron, paper_neuron, syn_spec={"synapse_model": "stdp_synapse_rec", "alpha": alpha, "weight" : w, "delay" : d, "tau_plus": tau_plus}) # May STDP on this one also

    for paper in graph.test_papers:
        for topic in graph.topics:
            paper_neuron = paper_neurons[paper]
            topic_neuron = topic_neurons[topic]
            w = config["test_to_topic_weight"]
            d = config["test_to_topic_delay"]
            alpha = config["test_to_topic_alpha"]
            tau_plus = config["test_to_topic_tau_plus"]
            nest.Connect(paper_neuron, topic_neuron, syn_spec={"synapse_model": "stdp_synapse_rec", "alpha": alpha, "weight" : w, "delay" : d, "tau_plus" : tau_plus}) # Need to figure out how to do STDP on this synapse with NEST
            nest.Connect(topic_neuron, paper_neuron, syn_spec={"synapse_model": "stdp_synapse_rec", "alpha": alpha, "weight" : w, "delay" : d, "tau_plus" : tau_plus}) # May STDP on this one also

    if (config["features"] == 1):
        for node in graph.graph.nodes:
            if (node not in graph.features.keys()):
                print("No features for node: ", node)
                continue
            for i in range(len(graph.features[node])):
                if (graph.features[node][i] == 1):
                    paper_neuron = paper_neurons[node]
                    feature_neuron = feature_neurons[i]
                    w = config["paper_to_feature_weight"]
                    d = config["paper_to_feature_delay"]
                    alpha = config["paper_to_feature_alpha"]
                    tau_plus = config["paper_to_feature_tau_plus"]
                    stdp_on = config["paper_to_feature_stdp"]

                    if (stdp_on == True):
                        nest.Connect(paper_neuron, feature_neuron, syn_spec={"synapse_model": "stdp_synapse", "alpha": alpha, "weight" : w, "delay" : d, "tau_plus" : tau_plus})
                        nest.Connect(feature_neuron, paper_neuron, syn_spec={"synapse_model": "stdp_synapse", "alpha": alpha, "weight" : w, "delay" : d, "tau_plus" : tau_plus})
                    else:
                        if (config["paper_to_feature_weighted"] == True):
                            w = config["paper_to_feature_weight"]*graph.paper_to_features[node]/(1.0*len(graph.features[node]))
                        nest.Connect(paper_neuron, feature_neuron, syn_spec={"weight": w, "delay": d})
                        if (config["paper_to_feature_weighted"] == True):
                            w = config["feature_to_paper_weight"]*graph.feature_to_papers[i]/(1.0*len(graph.features.keys()))
                        nest.Connect(feature_neuron, paper_neuron, syn_spec={"weight": w, "delay": d})
                    
    return paper_neurons, topic_neurons, feature_neurons, paper_voltmeters, topic_voltmeters, feature_voltmeters, wr, paper_recorders, topic_recorders

def test_paper(x):
    paper = x[0]    # paper ID of the paper you want to test
    graph = x[1]    # loaded graph
    config = x[2]   # config parameters
    import nest
    nest.set_verbosity("M_QUIET")
    paper_neurons, topic_neurons, feature_neurons, paper_voltmeters, topic_voltmeters, feature_voltmeters, weight_recorder, paper_recorders, topic_recorders = load_network(graph, config)
    
    spike_generator = nest.Create("spike_generator")
    nest.SetStatus(spike_generator, {'spike_times' : np.array([1.])})
    neuron = paper_neurons[paper]
    nest.Connect(spike_generator, neuron, syn_spec={'weight' : 100.})

    nest.Simulate(config["simtime"])

    potentials = []
    times = []
    topics = []   

    possible_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    topic_to_color = {}
    i = 0
    for topic in topic_neurons.keys():
        topic_to_color[topic] = possible_colors[i]
        i += 1

    i = 0
    if (config["monitors"] == True):
        for p in paper_recorders.keys():
            times = paper_recorders[p].get('events', 'times')
            if (p in graph.train_papers):
                s = '*'
            else:
                s = '.'
            s += topic_to_color[graph.paper_to_topic[p]]
            plt.plot(times, [i]*len(times), s)
            i += 1

        i = 0
        for t in topic_recorders.keys():
            times = topic_recorders[t].get('events', 'times')
            s = 'o' + topic_to_color[t]
            plt.plot(times, [i]*len(times), s)
            i += 1 

    paper_neuron = neuron
    paper_id = nest.GetStatus(paper_neuron, "global_id")[0]

    topic_ids = {} 
    times = []
    min_weight = 1000
    for topic in topic_neurons.keys():
        topic_neuron = topic_neurons[topic]
        if (config["monitors"] == True):
            #nest.voltage_trace.from_device(topic_voltmeters[topic])
            p = topic_voltmeters[topic].get('events', 'V_m')
            t = topic_voltmeters[topic].get('events', 'times')
            potentials.append(p)
            times.append(t)
            topics.append(topic)
        synapse = nest.GetConnections(topic_neuron, paper_neuron, "stdp_synapse_rec")
        w = synapse.get("weight")
        topic_id = nest.GetStatus(topic_neuron, "global_id")[0]
        topic_ids[topic] = topic_id
        print(topic, paper, w)
        if (w < min_weight):
            min_weight = w
            min_topic = topic

    retval = 0
    if (graph.paper_to_topic[paper] == min_topic):
        print("MIN VAL for ", paper, " Topic ", min_topic, " CORRECT")
        retval = 1
    else:
        print("MIN VAL for ", paper, " Topic ", min_topic, " WRONG ", graph.paper_to_topic[paper])
        retval = 0

    wr_weights = nest.GetStatus(weight_recorder, "events")[0]

    if (config["monitors"] == True):
        plt.figure()
        for i in range(len(potentials)):
            plt.plot(times[i], potentials[i], '-'+topic_to_color[topics[i]], label=topics[i])
        plt.plot(times[0], [config["topic_threshold"]]*len(times[0]), 'k-')
        plt.legend()
        plt.figure()

        for topic in topic_ids.keys():
            weights = [config["validation_to_topic_weight"]]
            times = [0.0]
            for i in range(len(wr_weights["senders"])):
                if (topic_ids[topic] == wr_weights["senders"][i] and paper_id == wr_weights["targets"][i]):
                    weights.append(wr_weights["weights"][i])
                    times.append(wr_weights["times"][i])
            plt.plot(times, weights, '-'+topic_to_color[topic], label=topic + "_weight")
        plt.legend()
                      
 
        plt.show()

    return retval
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GNN-SNN")
    parser.add_argument("--dataset", "-d", type=str, choices=["cora", "citeseer", "pubmed"], required=True)
    parser.add_argument("--mode", "-m", type=str, choices=["validation", "test"], required=True)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--features", type=int, choices=[0,1])
    parser.add_argument("--paper_leak", type=float, default=100000.0)           # [1., 10., 100., 100000.] 
    parser.add_argument("--paper_threshold", type=float, default=1.)            # [0.25, 0.5, 0.75, 1.0, 1.5]
    parser.add_argument("--paper_tau_minus", type=float, default=30.)           # [5., 10., 15., 20., 25., 30., 40., 50.]
    parser.add_argument("--train_ref", type=float, default=1.0)                 # [1.0, 5.0, 10.0, 100.0, 1000.0]
    parser.add_argument("--feature_ref", type=float, default=1.0)                 # [1.0, 5.0, 10.0, 100.0, 1000.0]
    parser.add_argument("--validation_ref", type=float, default=1000.0)         # [1.0, 5.0, 10.0, 100.0, 1000.0]

    # NOTE: We probably want to keep this parameter the same as the validation parameter above
    parser.add_argument("--test_ref", type=float, default=1000.0)               # [1.0, 5.0, 10.0, 100.0, 1000.0]
    parser.add_argument("--topic_leak", type=float, default=100000.0)           # [1., 10., 100., 100000.] 
    parser.add_argument("--topic_threshold", type=float, default=1.)            # [0.25, 0.5, 0.75, 1.0, 1.5]
    parser.add_argument("--topic_tau_minus", type=float, default=30.)           # [5., 10., 15., 20., 25., 30., 40., 50.]
    parser.add_argument("--feature_leak", type=float, default=100000.0)         # [1., 10., 100., 100000.] 
    parser.add_argument("--feature_threshold", type=float, default=1.0)         # [0.25, 0.5, 0.75, 1.0, 1.5]
    parser.add_argument("--feature_tau_minus", type=float, default=30.0)        # [5., 10., 15., 20., 25., 30., 40., 50.]
    parser.add_argument("--graph_weight", type=float, default=100.0)            # [0.5, 1.0, 10.0, 100.0]            
    parser.add_argument("--graph_delay", type=float, default=1.0)               # [1., 2., 5., 10., 20.]
    parser.add_argument("--train_to_topic_weight", type=float, default=1.0)     # [0.5, 1.0, 10.0, 100.0]
    parser.add_argument("--train_to_topic_delay", type=float, default=1.0)      # [1., 2., 5., 10., 20.]
    parser.add_argument("--validation_to_topic_weight", type=float, default=0.001)  # [0.001, 0.005, 0.01, 0.05, 0.1]
    parser.add_argument("--validation_to_topic_delay", type=float, default=1.0)     # [1., 2., 5., 10., 20.]
    parser.add_argument("--validation_to_topic_alpha", type=float, default=1.0)     # [0.5, 1.0, 2.0, 5.0]
    parser.add_argument("--validation_to_topic_tau_plus", type=float, default=30.0) # [5., 10., 15., 20., 25., 30., 40., 50.]

    # NOTE: We probably want to keep these parameters the same as the validation set parameters
    parser.add_argument("--test_to_topic_weight", type=float, default=0.001)    # [0.001, 0.005, 0.01, 0.05, 0.1]
    parser.add_argument("--test_to_topic_delay", type=float, default=1.0)       # [1., 2., 5., 10., 20.]
    parser.add_argument("--test_to_topic_alpha", type=float, default=1.0)       # [0.5, 1.0, 2.0, 5.0]
    parser.add_argument("--test_to_topic_tau_plus", type=float, default=30.0)   # [5., 10., 15., 20., 25., 30., 40., 50.]
    parser.add_argument("--paper_to_feature_weight", type=float, default=0.2)   # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    parser.add_argument("--feature_to_paper_weight", type=float, default=0.2)   # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    parser.add_argument("--paper_to_feature_delay", type=float, default=4.0)    # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    parser.add_argument("--paper_to_feature_alpha", type=float, default=1.0)    # [0.5, 1.0, 2.0, 5.0]
    parser.add_argument("--paper_to_feature_tau_plus", type=float, default=30.0)    # [5., 10., 15., 20., 25., 30., 40., 50.]
    parser.add_argument("--simtime", type=float, default=20.0)                  # [5., 10., 20., 30., 40., 50. 75., 100., 150., 200.]
    parser.add_argument("--processes", type=int, default=4)         # Change this depending on your machine
    parser.add_argument("--monitors", type=str, default="False")
    parser.add_argument("--paper_to_feature_stdp", type=str, default="False")
    parser.add_argument("--paper_to_feature_weighted", type=str, default="False")

    args = parser.parse_args()

    config = vars(args)

    if (config["monitors"] == "True" or config["monitors"] == "true"):
        config["monitors"] = True
    else:
        config["monitors"] = False

    if (config["paper_to_feature_stdp"] == "True" or config["paper_to_feature_stdp"] == "true"):
        config["paper_to_feature_stdp"] = True
    else:
        config["paper_to_feature_stdp"] = False
    
    if (config["paper_to_feature_weighted"] == "True" or config["paper_to_feature_weighted"] == "true"):
        config["paper_to_feature_weighted"] = True
    else:
        config["paper_to_feature_weighted"] = False

    np.random.seed(args.seed) 
    graph = GraphData(args.dataset, config)    
    
    if (config["monitors"] == True):
        for i in range(0, len(graph.validation_papers), 20):
            paper = graph.validation_papers[i]
            x = [paper, graph, config]
            test_paper(x)
        sys.exit()

     # Spike each of the test papers in simulation
    i = 0
    correct = 0
    total = 0

    pool = Pool(args.processes)
    if args.mode == "validation":
        papers = []
        for paper in graph.validation_papers:
            papers.append([paper, graph, config])
        x = pool.map(test_paper, papers)
        print("Validation Accuracy:", np.sum(x) / len(graph.validation_papers))
 
    if args.mode == "test":
        papers = []
        for paper in graph.test_papers:
            papers.append([paper, graph, config])
        x = pool.map(test_paper, papers) 
        print("Testing Accuracy:", np.sum(x) / len(graph.test_papers))
          

