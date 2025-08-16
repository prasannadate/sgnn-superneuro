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

        self.snn = snm.SNN()

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

    def configure_model(self):
        self.snn.apos = self.config["apos"]
        self.snn.aneg = self.config["aneg"]

    def make_network(self):
        model = self.snn
        # Read paper to paper edge list
        self.topic_neurons = topic_neurons = {}
        # Create paper neurons
        self.paper_neurons = paper_neurons = {}
        # create sets for faster __contains__ lookup
        train_papers = set(self.train_papers)
        validation_papers = set(self.validation_papers)
        test_papers = set(self.test_papers)

        c = self.config

        for node in self.graph.nodes:
            if (node not in self.paper_to_topic.keys()):
                continue
            if node in train_papers:
                neuron = model.create_neuron(threshold=c["paper_threshold"], leak=c["paper_leak"], refractory_period=c["train_ref"])
            elif node in validation_papers:
                neuron = model.create_neuron(threshold=c["paper_threshold"], leak=c["paper_leak"], refractory_period=c["validation_ref"])
            elif node in test_papers:
                neuron = model.create_neuron(threshold=c["paper_threshold"], leak=c["paper_leak"], refractory_period=c["test_ref"])
            paper_neurons[node] = neuron.idx

        for t in self.topics:
            neuron = model.create_neuron(threshold=c["topic_threshold"], leak=c["topic_leak"], refractory_period=0)
            topic_neurons[t] = neuron.idx

        for edge in self.graph.edges:
            paper, cited = edge
            if (paper not in self.paper_to_topic.keys() or cited not in self.paper_to_topic.keys()):
                continue
            pre = paper_neurons[paper]
            post = paper_neurons[cited]
            if pre == post:
                continue
            model.create_synapse(pre, post, weight=c["graph_weight"], delay=c["graph_delay"], stdp_enabled=False)
            model.create_synapse(post, pre, weight=c["graph_weight"], delay=c["graph_delay"], stdp_enabled=False)

        for paper in self.train_papers:
            paper_neuron = paper_neurons[paper]
            topic_neuron = topic_neurons[self.paper_to_topic[paper]]
            model.create_synapse(paper_neuron, topic_neuron, weight=c["train_to_topic_weight"], delay=c["train_to_topic_delay"], stdp_enabled=False)
            model.create_synapse(topic_neuron, paper_neuron, weight=c["train_to_topic_weight"], delay=c["train_to_topic_delay"], stdp_enabled=False)

        for paper in self.validation_papers:
            for topic in self.topics:
                paper_neuron = paper_neurons[paper]
                topic_neuron = topic_neurons[topic]
                model.create_synapse(paper_neuron, topic_neuron, stdp_enabled=True, weight=c["validation_to_topic_weight"], delay=c["validation_to_topic_delay"])
                model.create_synapse(topic_neuron, paper_neuron, stdp_enabled=True, weight=c["validation_to_topic_weight"], delay=c["validation_to_topic_delay"])

        for paper in self.test_papers:
            for topic in self.topics:
                paper_neuron = paper_neurons[paper]
                topic_neuron = topic_neurons[topic]
                model.create_synapse(paper_neuron, topic_neuron, stdp_enabled=True, weight=c["test_to_topic_weight"], delay=c["test_to_topic_delay"])
                model.create_synapse(topic_neuron, paper_neuron, stdp_enabled=True, weight=c["test_to_topic_weight"], delay=c["test_to_topic_delay"])


def test_paper_from_pickle(x):
    paper_id, temp = x
    with open(temp, 'rb') as f:
        d = pkl.load(f)
    return test_paper((paper_id, d))


def test_paper(x):
    paper_id, graph = x
    snn = graph.snn
    paper_neurons = graph.paper_neurons
    topic_neurons = graph.topic_neurons

    if (r := graph.resolution_order):
        # reorder topic_neurons by resolution order
        topic_neurons = {k: topic_neurons[k] for k in r}

    snn.add_spike(0, paper_neurons[paper_id], 100.0)
    # model.setup()
    snn.simulate(time_steps=graph.config["simtime"])

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
    ties = [topic for topic, weight in topic_weights if weight == best_weight]
    # if len(ties) > 1:  # check for ties
    #     best_topic = None  # don't count ties as correct
    actual_topic = graph.paper_to_topic[paper_id]
    total_spikes = snn.ispikes.sum()
    snn.release_mem()
    del snn, graph
    # match = actual_topic == best_topic
    ret = (actual_topic, ties)
    return ret, total_spikes


def score(answers, topics):
    tp, tn, fp, fn = 0, 0, 0, 0
    for actual, guesses in answers:
        for topic in topics:
            if topic == actual:
                if actual in guesses:
                    tp += 1
                else:
                    fn += 1
            else:
                if topic in guesses:
                    fp += 1
                else:
                    tn += 1
    return tp, tn, fp, fn


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

    print(f"Loaded config from {args.config}")
    print(f"This is the {config['dataset']} dataset over the {config['mode']} split.")

    backend = args.backend or config.get("backend", "auto")

    # create a random generator seeded with the config seed
    seed = config.get("seed", np.random.randint(0, 2**16))
    graph = GraphData(name=config["dataset"], config=config, rng=seed)
    graph.make_network()
    graph.configure_model()
    # topic_counts = graph.topic_prevalence()  # calculate topic counts
    # resolution_order = sorted(topic_counts, key=topic_counts.get, reverse=True)  # sort topics by prevalence
    # resolution_order = reversed(list(graph.topic_neurons))  # sort topics by reverse load order
    resolution_order = list(graph.topic_neurons)  # sort topics by load order
    graph.resolution_order = resolution_order

    if backend == "auto":
        backend = graph.snn.recommend(config["simtime"])
        graph.snn.backend = backend
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
    temp = tempfile.NamedTemporaryFile(delete=False)

    bundles = [(paper_id, temp.name) for paper_id in papers]
    with open(temp.name, 'wb') as f:
        pkl.dump(graph, f)
    with open('model.json', 'w') as f:
        graph.snn.saveas_json(f, array_representation="json-native")
    graph.snn.pretty_print(10)
    del papers, graph  # unload data and SNN to save memory

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
    # accuracy = np.sum(results) / n
    tp, tn, fp, fn = score(results, resolution_order)
    correct = np.sum([ans in guesses for ans, guesses in results])
    perfect = np.sum([set([ans]) == set(guesses) for ans, guesses in results])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"{mode.title()} results:")
    print(f"Recall:    {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"Positives: {correct} / {n} ({correct / n * 100:.2f}%)")
    print(f"Perfect:   {perfect} / {n} ({perfect / n * 100:.2f}%)")
    print(f"Total spikes:", np.sum(spikes))
    if config.get("dump_json", None):
        with open('results.json', 'a') as f:
            accuracy = {
                f"{mode}_accuracy": accuracy
            }
            dump_data = config | accuracy
            json.dump(dump_data, f, indent=2)
