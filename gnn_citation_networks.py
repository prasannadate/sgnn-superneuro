import os
import time
import json
import argparse
import tempfile
import warnings
import xyaml as yaml
import pickle as pkl
import pathlib as pl
from multiprocessing import Pool
from dataclasses import dataclass, field
from itertools import product

import tqdm
import numpy as np
import networkx as nx
import superneuromat as snm
# import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

wd = pl.Path(__file__).parent.absolute()  # get the Path() of this python file

defaultconfig_path = wd / 'configs/default_config.yaml'

default_config = yaml.load(open(defaultconfig_path, 'r'))


# from: https://stackoverflow.com/a/21894086/2712730
class bidict(dict):
    """Creates a dictionary that supports reverse lookups via the .inverse attribute.

    Args:
        dict (dict): The original dictionary.

    Properties:
        inverse (dict): A dictionary that maps values to keys.
    """
    def __new__(cls, *args, **kwargs):
        d = super().__new__(cls, *args, **kwargs)
        d.inverse = {}
        return d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super().__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super().__delitem__(key)


@dataclass
class Paper:
    idx: str  # Paper ID
    label: str = ''  # Paper category/topic
    features: tuple[bool | int | float, ...] = ()  # binary features
    citations: list[str] = field(default_factory=list)  # IDs of papers cited by this paper
    neuron: snm.Neuron = None

    # def __hash__(self):
    #     return hash(self.idx) + 2000

    def __str__(self):
        return self.idx


class GraphData():
    def __init__(self, name, config, rng=None):
        self.name = name
        self.config = config
        self.papers = {}
        self.paper_neurons = bidict()
        self.topic_neurons = bidict()
        self.topics = []            # the list of topics
        self.paper_featurecount = {}  # keyed on paper ID, value is the number of features it has
        self.feature_papercount = {}  # keyed on idx of feature in feature vector, value is the number of papers that have that feature
        self.feature_neurons = {}  # keyed on idx of feature in feature vector, value is the neuron ID of the feature neuron
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

        self.load_graph()
        self.load_papers()
        self.load_topics()
        # some papers appear as a citation but don't have a label. Remove those papers.
        missing = [paper.idx for paper in self.papers.values() if not paper.label]
        for paper in missing:
            del self.papers[paper]
        if self.config['features']:
            self.load_features()
        self.train_val_test_split()

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
                self.papers[paper_id].label = label
        elif self.nodes_path.suffix == ".tab":
            with open(self.nodes_path, 'r') as f:
                lines = f.readlines()  # read the file
            lines = lines[2:]  # skip the first two lines
            for line in lines:
                fields = line.strip().split()  # split on tab separator
                paper_idx, label, *_features = fields  # extract paper name and topic/label
                topics.add(label)
                paper_idx = int(paper_idx.strip().removeprefix("paper:"))  # make paper ID an int
                self.papers[paper_idx].label = label  # associate paper ID (as int) with topic/label
        self.topics = list(topics)

    def load_features(self):
        self.features = {}  # keyed on paper ID, value is the feature vector
        if self.nodes_path.suffix == ".content":
            with open(self.nodes_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                paper_id, *features, _label = line.strip().split()
                self.papers[paper_id].features = tuple(int(x) for x in features)
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
                paper_id, _topic, *features, summary = fields  # extract paper name and topic/label
                paper_id = int(paper_id.strip().removeprefix("paper:"))  # make paper ID an int
                self.papers[paper_id].features = [int(name in summary) for name in all_features]  # convert to binary features

        self.num_features = len(self.papers[paper_id].features)

        self.paper_featurecount = {}  # keyed on paper ID, value is the number of features it has
        self.feature_papercount = {}  # keyed on feature ID, value is the number of papers that have that feature
        for p in self.features.keys():
            self.paper_featurecount[p] = np.sum(self.features[p])
            for i in range(len(self.features[p])):
                if (i not in self.feature_papercount.keys()):
                    self.feature_papercount[i] = 0
                self.feature_papercount[i] += self.features[p][i]

    def load_graph(self):
        if self.edges_path.suffix == ".cites":
            self.graph = nx.read_edgelist(self.edges_path)
        elif self.edges_path.suffix == ".tab":
            self.graph = nx.from_edgelist(self.read_directed_cites_tab(self.edges_path))

    def load_papers(self):
        for paper in self.graph.nodes:
            self.papers[paper] = Paper(paper)

    def train_val_test_split(self):
        pool = list(self.papers)
        self.rng.shuffle(pool)

        # take & return the first `n` papers from the pool without replacement
        def select_papers(pool, n=1, removefrom=()):
            # bisect
            chosen, pool[:] = pool[0:n], pool[n:]
            # remove chosen papers from any auxiliary pools
            chosenset = set(chosen)
            for aux_pool in removefrom:
                aux_pool[:] = [paper for paper in aux_pool if paper not in chosenset]
            return chosen

        def select_topics_evenly(pool, n):
            chosen = []
            topic_papers = {topic: [] for topic in self.topics}

            def all_full():
                return all([len(topic_papers[topic]) == n for topic in self.topics])

            # make stacks of max size n for each topic
            for paper in pool:
                topic = self.papers[paper].label
                if len(topic_papers[topic]) < n:
                    topic_papers[topic].append(paper)
                    if all_full():
                        break

            # stack_sizes = [(len(papers), topic) for topic, papers in topic_papers.items()]
            # stack_sizes = sorted(stack_sizes, reverse=True)  # sort by size of topic stack

            # pick up to n / len(self.topics) papers from each stack
            for papers in topic_papers.values():
                selected = select_papers(papers, n // len(self.topics), removefrom=[pool])
                chosen.extend(selected)

            # pick the leftovers using a uniform distribution over topics as the stacks run out
            while len(chosen) < n:
                if not topic_papers:
                    raise RuntimeError("No papers left to select from.")
                topic = self.rng.choice(list(topic_papers))
                candidates = topic_papers[topic]
                if not candidates:
                    del topic_papers[topic]
                    continue
                chosen.extend(select_papers(candidates, 1, removefrom=[pool]))

            return chosen

        # take papers from pool and assign them to train, validation, and test sets
        # for train and validation, you can set `train_size` and `validation_size` in the config
        # or if that is `null` or not set, it will be set to `train_topics_mult` * `num_topics`
        if not (train_size := self.config.get("train_size", None)):
            train_size = self.config["train_topics_mult"] * len(self.topics)
        self.train_papers = select_topics_evenly(pool, train_size)

        if not (validation_size := self.config.get("validation_size", None)):
            validation_size = self.config["validation_topics_mult"] * len(self.topics)
        self.validation_papers = select_topics_evenly(pool, validation_size)

        # for the test set, you can set `test_size` in the config.
        # if that is `'auto'`, `null` or not set, it will be all of the remaining papers
        if not (test_size := self.config.get("test_size", None)) or test_size == 'auto':
            if not pool:
                warnings.warn(f"WARNING: TEST SET IS EMPTY.", RuntimeWarning, stacklevel=2)

            self.test_papers = pool  # remaining papers
        else:
            self.test_papers = pool[:test_size]

            if len(self.test_papers) < test_size:
                warnings.warn(f"WARNING: TEST SET IS SMALLER THAN EXPECTED. "
                            "Test set contains {len(self.test_papers)} papers.", RuntimeWarning, stacklevel=2)

    def topic_prevalence(self):
        topic_counts = {k: 0 for k in self.topics}
        for paper in self.papers.values():
            topic_counts[paper.label] += 1
        return topic_counts

    def configure_model(self):
        self.snn.apos = self.config["apos"]
        self.snn.aneg = self.config["aneg"]

    def make_network(self):
        model = self.snn
        # create sets for faster __contains__ lookup
        train_papers = set(self.train_papers)
        validation_papers = set(self.validation_papers)
        test_papers = set(self.test_papers)

        cfg = self.config
        # Create a neuron for each paper
        for paper in train_papers:
            self.paper_neurons[paper] = model.create_neuron(
                threshold=cfg["paper_threshold"], leak=cfg["paper_leak"], refractory_period=cfg["train_ref"])
        for paper in validation_papers:
            self.paper_neurons[paper] = model.create_neuron(
                threshold=cfg["paper_threshold"], leak=cfg["paper_leak"], refractory_period=cfg["validation_ref"])
        for paper in test_papers:
            self.paper_neurons[paper] = model.create_neuron(
                threshold=cfg["paper_threshold"], leak=cfg["paper_leak"], refractory_period=cfg["test_ref"])

        # Create a neuron for each topic
        for t in self.topics:
            neuron = model.create_neuron(threshold=cfg["topic_threshold"], leak=cfg["topic_leak"], refractory_period=0)
            self.topic_neurons[t] = neuron

        # Create bi-directional synapse for each edge in the graph
        for edge in self.graph.edges:
            paper, cited = edge
            if paper not in self.paper_neurons or cited not in self.paper_neurons:
                continue
            pre = self.paper_neurons[paper]
            post = self.paper_neurons[cited]
            if pre == post:
                continue
            model.create_synapse(pre, post, weight=cfg["graph_weight"], delay=cfg["graph_delay"], stdp_enabled=False)
            model.create_synapse(post, pre, weight=cfg["graph_weight"], delay=cfg["graph_delay"], stdp_enabled=False)

        for paper in self.train_papers:
            p = self.paper_neurons[paper]
            t = self.topic_neurons[self.papers[paper].label]
            model.create_synapse(p, t, weight=cfg["train_to_topic_weight"], delay=cfg["train_to_topic_delay"], stdp_enabled=False)
            model.create_synapse(t, p, weight=cfg["train_to_topic_weight"], delay=cfg["train_to_topic_delay"], stdp_enabled=False)

        for paper in self.validation_papers:
            for topic in self.topics:
                p = self.paper_neurons[paper]
                t = self.topic_neurons[topic]
                model.create_synapse(p, t, stdp_enabled=True, weight=cfg["validation_to_topic_weight"], delay=cfg["validation_to_topic_delay"])
                model.create_synapse(t, p, stdp_enabled=True, weight=cfg["validation_to_topic_weight"], delay=cfg["validation_to_topic_delay"])

        for paper in self.test_papers:
            for topic in self.topics:
                p = self.paper_neurons[paper]
                t = self.topic_neurons[topic]
                model.create_synapse(p, t, stdp_enabled=True, weight=cfg["test_to_topic_weight"], delay=cfg["test_to_topic_delay"])
                model.create_synapse(t, p, stdp_enabled=True, weight=cfg["test_to_topic_weight"], delay=cfg["test_to_topic_delay"])

        if not cfg['features']:
            return

        # connect features to paper neurons if features are enabled
        for feature_idx in range(self.num_features):
            feature = model.create_neuron(threshold=cfg["feature_threshold"], leak=cfg["feature_leak"], refractory_period=cfg["feature_ref"])
            self.feature_neurons[feature_idx] = feature.idx

        for neuron in self.paper_neurons:
            paper = self.paper_neurons.inverse[neuron]
            p = neuron
            features = self.features[paper]
            indices = np.nonzero(features)[0]
            for feature_idx in indices:
                f = self.feature_neurons[feature_idx]
                model.create_synapse(p, f, weight=cfg["paper_to_feature_weight"], delay=cfg["paper_to_feature_delay"], stdp_enabled=cfg['paper_to_feature_stdp'])
                model.create_synapse(f, p, weight=cfg["feature_to_paper_weight"], delay=cfg["paper_to_feature_delay"], stdp_enabled=cfg['paper_to_feature_stdp'])

        if not cfg['feature_to_topic']:
            return

        for feature, topic in product(self.feature_neurons.values(), self.topic_neurons.values()):
            model.create_synapse(feature, topic, weight=cfg["feature_to_topic_weight"], delay=cfg["feature_to_topic_delay"], stdp_enabled=cfg['feature_to_topic_stdp'])
            model.create_synapse(topic, feature, weight=cfg["feature_to_topic_weight"], delay=cfg["feature_to_topic_delay"], stdp_enabled=cfg['feature_to_topic_stdp'])


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
    actual_topic = graph.papers[paper_id].label
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
    # parser.add_argument('--config', default=wd / 'configs/miniseer/default_miniseer_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/citeseer/default_citeseer_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/cora/default_cora_config.yaml')
    parser.add_argument('--config', default=wd / 'configs/pubmed/default_pubmed_config.yaml')
    args = parser.parse_args()

    model_time = time.time()
    config = default_config.copy()
    with open(args.config, 'r') as f:
        config.update(yaml.load(f))

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
