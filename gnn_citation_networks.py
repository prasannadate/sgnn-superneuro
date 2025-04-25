import os
import time
import json
import argparse
import tempfile
import xyaml as yaml
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
    def __init__(self, name, config, rng=None):
        self.name = name
        self.config = config
        self.paper_to_topic = {}  # maps the paper ID in the dataset to its topic ID
        self.index_to_paper = []    # creates an index for each paper
        self.topics = []            # the list of topics
        self.resolution_order = []
        self.edges_path = pl.Path(config["edges_path"])
        self.nodes_path = pl.Path(config["nodes_path"])
        self.train_papers = []
        self.validation_papers = []
        self.test_papers = []

        if rng is None:
            self.seed = np.random.randint(0, 2**16)
            self.rng = np.random.default_rng(self.seed)
        elif not isinstance(rng, np.random.Generator):
            self.seed = rng
            self.rng = np.random.default_rng(self.seed)
        else:
            self.seed = None
            self.rng = rng

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
        topics = set()
        if self.nodes_path.suffix == ".content":
            with open(self.nodes_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                paper_id, *_features, label = line.strip().split()
                topics.add(label)
                self.paper_to_topic[paper_id] = label
                self.index_to_paper.append(paper_id)
        elif self.nodes_path.suffix == ".tab":
            with open(self.nodes_path, 'r') as f:
                lines = f.readlines()  # read the file
            lines = lines[2:]  # skip the first two lines
            for line in lines:
                fields = line.strip().split()  # split on tab separator
                paper_idx, topic, *_features = fields  # extract paper name and topic/label
                topics.add(topic)
                paper_idx = int(paper_idx.strip().removeprefix("paper:"))  # make paper ID an int
                self.paper_to_topic[paper_idx] = topic  # associate paper ID (as int) with topic/label
                self.index_to_paper.append(paper_idx)
        self.topics = list(topics)

    def load_features(self):
        self.features = {}  # keyed on paper ID, value is the feature vector
        if self.nodes_path.suffix == ".content":
            with open(self.nodes_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                paper_id, *features, _label = line.strip().split()
                self.features[paper_id] = [int(x) for x in features]
        elif self.nodes_path.suffix == ".tab":
            with open(self.nodes_path, 'r') as f:
                lines = f.readlines()  # read the file
            # get feature names
            feature_line = lines[1]
            fields = feature_line.split()
            _cat, *feature_names, _summary = fields
            all_features = [feature_desc.split(':') for feature_desc in feature_names]  # split each
            all_features = [name for _type, name, _val in all_features]  # get just the feature name
            # parse features for each paper
            lines = lines[2:]  # skip the first two lines
            for line in lines:
                fields = line.strip().split()  # split on tab separator
                paper_idx, _topic, *features, summary = fields  # extract paper name and topic/label
                paper_idx = int(paper_idx.strip().removeprefix("paper:"))  # make paper ID an int
                self.features[paper_idx] = [int(name in summary) for name in all_features]  # convert to binary features

        self.paper_to_features = {}  # keyed on paper ID, value is the number of features it has
        self.feature_to_papers = {}  # keyed on feature ID, value is the number of papers that have that feature
        for p in self.features.keys():
            self.paper_to_features[p] = np.sum(self.features[p])
            for i in range(len(self.features[p])):
                if (i not in self.feature_to_papers.keys()):
                    self.feature_to_papers[i] = 0
                self.feature_to_papers[i] += self.features[p][i]

    def load_graph(self):
        if self.edges_path.suffix == ".cites":
            self.graph = nx.read_edgelist(self.edges_path)
        elif self.edges_path.suffix == ".tab":
            self.graph = nx.from_edgelist(self.read_directed_cites_tab(self.edges_path))

    def train_val_test_split(self):
        train_papers = []
        test_papers = []
        validation_papers = []
        check_breakdown = {}

        for k in self.topics:
            check_breakdown[k] = 0

        while (len(train_papers) < len(self.topics) * 20):
            index = self.rng.integers(len(self.index_to_paper))
            topic = self.paper_to_topic[self.index_to_paper[index]]
            if (check_breakdown[topic] < 20 and self.index_to_paper[index] not in train_papers):
                train_papers.append(self.index_to_paper[index])
                check_breakdown[topic] += 1

        while (len(validation_papers) < len(self.topics) * 20):
            index = self.rng.integers(len(self.index_to_paper))
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
        train_papers = []
        test_papers = []
        validation_papers = []
        check_breakdown = {}

        for k in self.topics:
            check_breakdown[k] = 0

        while (len(train_papers) < len(self.topics) * 5):
            paper_id = self.rng.choice(self.index_to_paper)
            topic = self.paper_to_topic[paper_id]
            if (check_breakdown[topic] < 10 and paper_id not in train_papers):
                train_papers.append(paper_id)
                check_breakdown[topic] += 1

        while (len(validation_papers) < len(self.topics) * 5):
            paper_id = self.rng.choice(self.index_to_paper)
            topic = self.paper_to_topic[paper_id]
            if (check_breakdown[topic] < 20 and paper_id not in train_papers and paper_id not in validation_papers):
                validation_papers.append(paper_id)
                check_breakdown[topic] += 1

        for paper_id in self.index_to_paper:
            if (paper_id not in train_papers and paper_id not in validation_papers and paper_id in self.paper_to_topic.keys()):
                test_papers.append(paper_id)

        self.train_papers = train_papers
        self.test_papers = test_papers
        self.validation_papers = validation_papers

    def topic_prevalence(self):
        topic_counts = {k: 0 for k in self.topics}
        for topic in self.paper_to_topic.values():
            topic_counts[topic] += 1
        return topic_counts


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
        paper, cited = edge
        if (paper not in graph.paper_to_topic.keys() or cited not in graph.paper_to_topic.keys()):
            continue
        pre = paper_neurons[paper]
        post = paper_neurons[cited]
        if pre == post:
            continue
        graph_weight = 100.0
        variations = (1.0 + graph.rng.normal(0, variation_scale))
        graph_delay = 1
        model.create_synapse(pre, post, weight=config["graph_weight"], delay=config["graph_delay"], stdp_enabled=False)
        model.create_synapse(post, pre, weight=config["graph_weight"], delay=config["graph_delay"], stdp_enabled=False)

    for paper in graph.train_papers:
        paper_neuron = paper_neurons[paper]
        topic_neuron = topic_neurons[graph.paper_to_topic[paper]]
        train_to_topic_w = 1.0
        train_to_topic_w *= (1.0 + graph.rng.normal(0, variation_scale))
        train_to_topic_d = 1
        model.create_synapse(paper_neuron, topic_neuron, weight=config["train_to_topic_weight"], delay=config["train_to_topic_delay"], stdp_enabled=False)
        model.create_synapse(topic_neuron, paper_neuron, weight=config["train_to_topic_weight"], delay=config["train_to_topic_delay"], stdp_enabled=False)

    for paper in graph.validation_papers:
        for topic in graph.topics:
            paper_neuron = paper_neurons[paper]
            topic_neuron = topic_neurons[topic]

            validation_to_topic_w = 0.001
            validation_to_topic_w *= (1.0 + graph.rng.normal(0, variation_scale))
            validation_to_topic_d = 1
            model.create_synapse(paper_neuron, topic_neuron, stdp_enabled=True, weight=config["validation_to_topic_weight"], delay=config["validation_to_topic_delay"])
            model.create_synapse(topic_neuron, paper_neuron, stdp_enabled=True, weight=config["validation_to_topic_weight"], delay=config["validation_to_topic_delay"])

    for paper in graph.test_papers:
        for topic in graph.topics:
            paper_neuron = paper_neurons[paper]
            topic_neuron = topic_neurons[topic]
            test_to_topic_w = 0.001
            test_to_topic_w *= (1.0 + graph.rng.normal(0, variation_scale))
            test_to_topic_d = 1
            model.create_synapse(paper_neuron, topic_neuron, stdp_enabled=True, weight=config["test_to_topic_weight"], delay=config["test_to_topic_delay"])
            model.create_synapse(topic_neuron, paper_neuron, stdp_enabled=True, weight=config["test_to_topic_weight"], delay=config["test_to_topic_delay"])

    return paper_neurons, topic_neurons, model


def create_model(graph, config):
    paper_neurons, topic_neurons, model = load_network(graph, config)
    model.apos = config["apos"]
    model.aneg = config["aneg"]
    model.backend = config.get("backend", "auto")
    return model, paper_neurons, topic_neurons


def test_paper_from_pickle(x):
    paper_id, temp = x
    with open(temp, 'rb') as f:
        d = pkl.load(f)
    return test_paper((paper_id, d))


# def get_synapse(model, pre_id, post_id):
#     for i, (pre, post) in enumerate(zip(model.pre_synaptic_neuron_ids, model.post_synaptic_neuron_ids)):
#         if pre_id == pre and post_id == post:
#             return i


def test_paper(x):
    paper_id, d = x
    snn = d["model"]
    graph = d["graph"]
    paper_neurons = d["paper_neurons"]
    topic_neurons = d["topic_neurons"]
    config = d["config"]

    if (r := graph.resolution_order):
        # reorder topic_neurons by resolution order
        topic_neurons = {k: topic_neurons[k] for k in r}

    snn.add_spike(0, paper_neurons[paper_id], 100.0)
    # model.setup()
    snn.simulate(time_steps=config["simtime"])

    # Analyze the weights between the test paper neuron and topic neurons
    topic_weights = []
    for topic_id, topic_paper_id in topic_neurons.items():
        # Find the synapse from topic neuron to test paper neuron
        synapse = snn.get_synapse(paper_neurons[paper_id], topic_paper_id)
        topic_weights.append((topic_id, synapse.weight))

    # sort by highest to lowest weight
    # ties are resolved by the order of topic_weights, which is ordered by topic_neurons
    topic_weights = sorted(topic_weights, key=lambda x: x[1], reverse=True)
    best_topic, best_weight = topic_weights[0]
    # ties = [topic for topic, weight in topic_weights if weight == best_weight]
    # if len(ties) > 1:  # check for ties
    #     best_topic = None  # don't count ties as correct
    actual_topic = graph.paper_to_topic[paper_id]
    total_spikes = snn.ispikes.sum()
    snn.release_mem()
    del snn, graph
    del d
    match = actual_topic == best_topic
    return match, total_spikes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=["auto", "cpu", "jit", "gpu"])
    parser.add_argument('--config', default=wd / 'configs/miniseer/default_miniseer_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/citeseer/default_citeseer_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/cora/default_cora_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/pubmed/default_pubmed_config.yaml')
    args = parser.parse_args()

    model_time = time.time()
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    backend = args.backend or config.get("backend", "auto")

    # create a random generator seeded with the config seed
    seed = config.get("seed", np.random.randint(0, 2**16))
    graph = GraphData(config["dataset"], config, rng=seed)
    topic_counts = graph.topic_prevalence()  # sort topics by prevalence
    graph.resolution_order = sorted(topic_counts, key=topic_counts.get, reverse=True)
    model, paper_neurons, topic_neurons = create_model(graph, config)

    if backend == "auto":
        backend = model.recommend(config["simtime"])
        model.backend = backend
    if backend == "gpu":
        processes = config.get("gpu_processes", 1)
    elif config.get("processes", "auto") == "auto":
        processes = os.cpu_count()
    else:
        processes = config["processes"]

    papers = []
    mode = config['mode']
    if mode == 'test':
        papers = graph.test_papers
    if mode == "validation":
        papers = graph.validation_papers
    n = len(papers)

    model_time = time.time() - model_time
    print(f"Time to load dataset and create model: {model_time} seconds")
    # save model to file for multiprocessing
    d = {
        'config': config,
        'model': model,
        'graph': graph,
        'paper_neurons': paper_neurons,
        'topic_neurons': topic_neurons,
    }
    temp = tempfile.NamedTemporaryFile(delete=False)

    bundles = [(paper_id, temp.name) for paper_id in papers]
    with open(temp.name, 'wb') as f:
        pkl.dump(d, f)
    del papers, model, graph, paper_neurons, topic_neurons  # unload to save memory

    load_time = time.time()
    with open(temp.name, 'rb') as f:
        d = pkl.load(f)
    del d
    load_time = time.time() - load_time
    print(f"Time to load model from pickle: {load_time} seconds")

    eval_time = time.time()

    if processes == 1:
        # single-process evaluation
        x = [test_paper_from_pickle(bundle) for bundle in tqdm.tqdm(bundles)]
    else:
        try:  # multi-process evaluation
            x = process_map(test_paper_from_pickle, bundles, max_workers=processes, chunksize=1)  # with tqdm
            # pool = Pool(2)
            # x = pool.map(test_paper, bundles)
            pass
        finally:  # clean up and delete the temp file no matter what
            temp.close()  # close the temp file
            os.unlink(temp.name)  # DON'T COMMENT OUT THIS.
    eval_time = time.time() - eval_time
    print(f"Evaluation time: {eval_time} seconds")

    results, spikes = zip(*x)
    accuracy = np.sum(results) / n
    print(f"{mode.title()} Accuracy: {accuracy}")
    print(f"Correct: {np.sum(results)} / {n}")
    print(f"Total spikes:", np.sum(spikes))
    if config.get("dump_json", None):
        with open('results.json', 'a') as f:
            accuracy = {
                f"{mode}_accuracy": accuracy
            }
            dump_data = config | accuracy
            json.dump(dump_data, f, indent=2)
