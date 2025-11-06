import os
import time
import json
import argparse
import tempfile
import warnings
import xyaml as yaml
import pickle as pkl
import pathlib as pl
from itertools import product
from collections import Counter
from multiprocessing import Pool
from dataclasses import dataclass, field

import tqdm
import numpy as np
import networkx as nx
import superneuromat as snm
# import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
from sklearn.preprocessing import MultiLabelBinarizer

# typing:
from superneuromat import Neuron, Synapse

Pname = str | int

wd = pl.Path(__file__).parent.absolute()  # get the Path() of this python file

defaultconfig_path = wd / 'configs/default_config.yaml'

# this file is used to set the base config
# the per-dataset config is merged into this later
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
    features: tuple[bool | int | float, ...] | list[bool | int | float]
    features = ()  # binary features
    citations: list[str] = field(default_factory=list)  # IDs of papers cited by this paper
    neuron: snm.Neuron = None

    # def __hash__(self):
    #     return hash(self.idx) + 2000

    def __str__(self):
        return str(self.idx)


class GraphData():
    def __init__(self, name, config, **kwargs):
        self.name = name
        config.update(kwargs)
        self.config = config
        self.papers: dict[Pname, Paper] = {}
        self.topics: list[str] = []  # the list of topics
        self.resolution_order: list[Pname] = []
        self.edges_path = pl.Path(self.config["edges_path"])
        self.nodes_path = pl.Path(self.config["nodes_path"])
        self.train_papers: list[Pname] = []
        self.validation_papers: list[Pname] = []
        self.test_papers: list[Pname] = []
        self.mlb = MultiLabelBinarizer()

        self.seed = config.get("seed", None)

    def load_all(self, remove_missing=True):
        self.load_graph()
        self.load_papers()
        self.load_topics()
        # some papers appear as a citation but don't have a label. Remove those papers.
        if remove_missing:
            missing = [paper.idx for paper in self.papers.values() if not paper.label]
            for paper in missing:
                del self.papers[paper]
        if self.config['features']:
            self.load_features()
        if self.config['legacy_split']:
            self.train_val_test_split_legacy()
        else:
            self.train_val_test_split()
        self.select_papers(self.config['mode'])

    def select_papers(self, mode):
        if mode == 'test':
            self.selected_papers = self.test_papers
        elif mode == "validation":
            self.selected_papers = self.validation_papers
        elif mode == "train":
            self.selected_papers = self.train_papers
        else:
            self.selected_papers = []

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, rng: int | np.random.Generator | None):
        if rng is None:
            self._seed = np.random.randint(0, 2**16)
            self.rng = np.random.default_rng(self._seed)
        elif not isinstance(rng, np.random.Generator):
            self._seed = rng
            self.rng = np.random.default_rng(self._seed)
        else:
            self._seed = None
            self.rng = rng

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
        self.mlb = MultiLabelBinarizer(classes=self.topics)

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

    def train_val_test_split_legacy(self):
        if (pick_evenly := self.config.get('pick_evenly', None)) is not None:
            pick_evenly = set(pick_evenly) - {'null', None}
            if pick_evenly != {'train', 'validation'}:
                warnings.warn("WARNING: You're using config['legacy_split'] which only supports "
                              "config['pick_evenly'] = ['train', 'validation']. "
                              "We'll ignore your 'pick_evenly' setting.", RuntimeWarning, 2)

        paperlist = list(self.papers.values())

        def select_papers(topics_mult=20, used: set[int | str] | None = None):
            used = used or set()  # papers that should not be selected
            selected = []
            topic_counts = Counter()
            attempts = 0

            while len(selected) < len(self.topics) * topics_mult:
                idx = self.rng.integers(len(self.papers))
                paper = paperlist[idx]
                if (
                    topic_counts[paper.label] < topics_mult
                    and paper not in selected
                    and paper.idx not in used
                ):
                    selected.append(paper.idx)
                    topic_counts[paper.label] += 1
                    attempts = 0
                else:
                    attempts += 1
                    if attempts > 1000:
                        raise RuntimeError("Failed to select a paper due to bad luck or lack of valid papers.")

            return selected

        self.train_papers = select_papers(topics_mult=self.config["train_topics_mult"])
        self.validation_papers = select_papers(used=set(self.train_papers),  # disallow papers in the train set
                                               topics_mult=self.config["validation_topics_mult"])
        test_set = set(self.papers) - set(self.train_papers + self.validation_papers)  # remaining papers
        self.test_papers = [papername for papername in self.papers if papername in test_set]  # preserve order

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

        class Selector:
            @classmethod
            def __getitem__(cls, name):
                if name in self.config.get("pick_evenly", []):
                    return select_topics_evenly
                return select_papers
        select_func = Selector()

        # take papers from pool and assign them to train, validation, and test sets
        # for train and validation, you can set `train_size` and `validation_size` in the config
        # or if that is `null` or not set, it will be set to `train_topics_mult` * `num_topics`
        if not (train_size := self.config.get("train_size", None)):
            train_size = self.config["train_topics_mult"] * len(self.topics)
        self.train_papers = select_func['train'](pool, train_size)

        if not (validation_size := self.config.get("validation_size", None)):
            validation_size = self.config["validation_topics_mult"] * len(self.topics)
        self.validation_papers = select_func['test'](pool, validation_size)

        # for the test set, you can set `test_size` in the config.
        # if that is `'auto'`, `null` or not set, it will be all of the remaining papers
        if not (test_size := self.config.get("test_size", None)) or test_size == 'auto':
            if not pool:
                warnings.warn(f"WARNING: TEST SET IS EMPTY.", RuntimeWarning, stacklevel=2)

            self.test_papers = pool  # remaining papers
        else:
            self.test_papers = select_func['test'](pool, test_size)

            if len(self.test_papers) < test_size:
                warnings.warn(f"WARNING: TEST SET IS SMALLER THAN EXPECTED. "
                            "Test set contains {len(self.test_papers)} papers.", RuntimeWarning, stacklevel=2)

    def topic_prevalence(self, papers=None):
        if papers is None:
            papers = self.papers.values()
        topic_counts = {k: 0 for k in self.topics}  # create a dictionary with all topics as keys
        for paper in papers:
            label = paper.label if isinstance(paper, Paper) else self.papers[paper].label
            topic_counts[label] += 1
        return topic_counts

    def topic_breakdowns(self):
        import pandas as pd
        all_papers = self.topic_prevalence()
        test = self.topic_prevalence(self.test_papers)
        train = self.topic_prevalence(self.train_papers)
        validation = self.topic_prevalence(self.validation_papers)
        df = pd.DataFrame([all_papers, test, train, validation], index=['all', 'test', 'train', 'validation'])
        return df


class SGNN(GraphData):
    def __init__(self, name, config, **kwargs):
        super().__init__(name, config, **kwargs)
        self.paper_neurons: bidict[Pname, Neuron] = bidict()
        self.topic_neurons: bidict[str, Neuron] = bidict()
        self.feature_neurons = {}  # keyed on idx of feature in feature vector, value is the neuron ID of the feature neuron
        self.snn = snm.SNN()

    def mp_processes(self, override=None):
        backend = override or self.config.get("backend", "auto")
        if backend == "auto":
            backend = self.snn.recommend(self.config["simtime"])
        if backend == "gpu":
            processes = int(self.config.get("gpu_processes", 1))
        elif self.config.get("processes", "auto") == "auto":
            processes = os.cpu_count()
            if processes is None:
                raise RuntimeError("Unable to determine number of CPUs. Please specify the number of processes manually.")
        else:
            processes = self.config["processes"]
        self.snn.backend = backend
        return processes

    def make_network(self):
        model = self.snn
        # create sets for faster __contains__ lookup
        train_papers = set(self.train_papers)
        validation_papers = set(self.validation_papers)
        test_papers = set(self.test_papers)

        # set the apos and aneg values for STDP
        self.snn.apos = self.config["apos"]
        self.snn.aneg = self.config["aneg"]

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
            model.create_synapse(pre, post, weight=cfg["graph_weight"], delay=cfg["graph_delay"])
            model.create_synapse(post, pre, weight=cfg["graph_weight"], delay=cfg["graph_delay"])

        for paper in self.train_papers:
            p = self.paper_neurons[paper]
            t = self.topic_neurons[self.papers[paper].label]
            model.create_synapse(p, t, weight=cfg["train_to_topic_weight"], delay=cfg["train_to_topic_delay"])
            model.create_synapse(t, p, weight=cfg["train_to_topic_weight"], delay=cfg["train_to_topic_delay"])

        for paper in self.validation_papers:
            for topic in self.topics:
                p = self.paper_neurons[paper]
                t = self.topic_neurons[topic]
                model.create_synapse(p, t, weight=cfg["validation_to_topic_weight"], delay=cfg["validation_to_topic_delay"], stdp_enabled=True)
                model.create_synapse(t, p, weight=cfg["validation_to_topic_weight"], delay=cfg["validation_to_topic_delay"], stdp_enabled=True)

        for paper in self.test_papers:
            for topic in self.topics:
                p = self.paper_neurons[paper]
                t = self.topic_neurons[topic]
                model.create_synapse(p, t, weight=cfg["test_to_topic_weight"], delay=cfg["test_to_topic_delay"], stdp_enabled=True)
                model.create_synapse(t, p, weight=cfg["test_to_topic_weight"], delay=cfg["test_to_topic_delay"], stdp_enabled=True)

        if not cfg['features']:
            return

        # connect features to paper neurons if features are enabled
        for feature_idx in range(self.num_features):
            feature = model.create_neuron(threshold=cfg["feature_threshold"], leak=cfg["feature_leak"], refractory_period=cfg["feature_ref"])
            self.feature_neurons[feature_idx] = feature.idx

        for paper_idx, neuron in self.paper_neurons.items():
            p = neuron
            features = self.papers[paper_idx].features
            indices = np.nonzero(features)[0]  # pyright: ignore[reportArgumentType]
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
    graph: SGNN
    snn = graph.snn
    topic_neurons = graph.topic_neurons
    paper = graph.papers[paper_id]
    paper_neuron = graph.paper_neurons[paper_id]

    if (r := graph.resolution_order):
        # reorder topic_neurons by resolution order
        topic_neurons = {k: topic_neurons[k] for k in r}

    snn.add_spike(0, paper_neuron, 100.0)
    snn.simulate(time_steps=graph.config["simtime"])

    # Analyze the weights between the test paper neuron and topic neurons
    topic_weights = []
    for topic_id, topic_paper_id in topic_neurons.items():
        # Find the synapse from topic neuron to test paper neuron
        synapse = snn.get_synapse(paper_neuron, topic_paper_id)
        topic_weights.append((topic_id, synapse.weight))

    # sort by highest to lowest weight
    # ties are resolved by the order of topic_weights, which is ordered by topic_neurons
    topic_weights = sorted(topic_weights, key=lambda x: x[1], reverse=True)
    _best_topic, best_weight = topic_weights[0]
    ties = [topic for topic, weight in topic_weights if weight == best_weight]
    # if len(ties) > 1:  # check for ties
    #     best_topic = None  # don't count ties as correct

    # if there are still ties, narrow down further by looking at paper->feature->topic weights
    if graph.config['features'] and len(ties) > 1:
        topic_scores = []
        for topic in ties:
            # score each topic by summing the weights of
            # paper -> feature[i] -> topic synapses over i, if feature[i] is active
            feature_neurons = [graph.feature_neurons[i] for i in np.nonzero(paper.features)[0]]  # pyright: ignore[reportArgumentType]
            score = 0
            for feature_neuron in feature_neurons:
                paper_to_feature = snn.get_synapse(paper_neuron, feature_neuron)
                feature_to_topic = snn.get_synapse(feature_neuron, topic_neurons[topic])
                score += paper_to_feature.weight * feature_to_topic.weight
            topic_scores.append((topic, score))
        # choose papers with the highest score
        _best_topic, max_score = max(topic_scores, key=lambda x: x[1])
        ties = [topic for topic, score in topic_scores if score == max_score]

    actual_topic = paper.label
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


def make_graph(args, base_config=default_config):
    config = base_config.copy()
    with open(args.config, 'r') as f:
        config.update(yaml.load(f))  # this config overrides entries in the default config

    print(f"Loaded dataset-specific config from {args.config}")
    print(f"This is the {config['dataset']} dataset over the {config['mode']} split.")

    # create a random generator seeded with the config seed
    graph = SGNN(name=config["dataset"], config=config)
    graph.load_all()
    graph.make_network()

    return graph


def evaluate(bundles, processes=1, temp=None):
    # evaluate the model for each paper
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
    return x


@dataclass
class Results:
    n: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    correct: int = 0
    perfect: int = 0
    spikes: tuple[np.ndarray, ...] = tuple()
    name: str = ''

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self):
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def __str__(self):
        return '\n'.join([
            f"{self.name} results:" if self.name else "Results:",
            f"Recall:    {self.recall:.3f}",
            f"Precision: {self.precision:.3f}",
            f"Accuracy:  {self.accuracy:.3f}",
            f"F1:        {self.f1:.3f}",
            f"Positives: {self.correct} / {self.n} ({self.correct / self.n * 100:.2f}%)",
            f"Perfect:   {self.perfect} / {self.n} ({self.perfect / self.n * 100:.2f}%)",
            f"Total spikes: {np.sum(self.spikes)}",
        ])


def calculate_accuracy(results, resolution_order, name=''):
    n = len(results)
    results, spikes = zip(*results)
    # accuracy = np.sum(results) / n
    tp, tn, fp, fn = score(results, resolution_order)
    correct = np.sum([ans in guesses for ans, guesses in results])
    perfect = np.sum([set([ans]) == set(guesses) for ans, guesses in results])
    return Results(n=n, tp=tp, tn=tn, fp=fp, fn=fn, correct=correct, perfect=perfect, spikes=spikes, name=name)


def main(args):

    model_time = time.time()

    graph = make_graph(args)
    config = graph.config
    processes = graph.mp_processes(args.backend)
    papers = graph.selected_papers
    mode = config['mode']
    if config['test_only_first']:
        papers = papers[:config['test_only_first']]

    # if there's a tie among topics, choose the topic closest to [0]
    # resolution_order = sorted(topic_counts, key=topic_counts.get, reverse=True)  # sort topics by prevalence
    # resolution_order = reversed(list(graph.topic_neurons))  # sort topics by reverse load order
    resolution_order = list(graph.topic_neurons)  # sort topics by load order
    graph.resolution_order = resolution_order

    model_time = time.time() - model_time
    print(f"Time to load dataset and create model: {model_time} seconds")
    # save model to file for multiprocessing
    temp = tempfile.NamedTemporaryFile(delete=False)
    with open(temp.name, 'wb') as f:
        pkl.dump(graph, f)
    # with open('model.json', 'w') as f:
    #     graph.snn.saveas_json(f, array_representation="json-native")
    # create a list of (paper_id, pickled_file) tuples for multiprocessing
    bundles = [(paper_id, temp.name) for paper_id in papers]

    print(graph.snn.pretty(10))
    print("Topic breakdowns:")
    print(graph.topic_breakdowns())
    del papers, graph  # unload data and SNN to save memory

    # test loading the model from pickle
    load_time = time.time()
    with open(temp.name, 'rb') as f:
        d = pkl.load(f)
    del d
    load_time = time.time() - load_time
    print(f"Time to load model from pickle: {load_time} seconds")

    x = evaluate(bundles, processes, temp)

    results = calculate_accuracy(x, resolution_order, mode.title())
    print(results)

    if config.get("dump_json", None):
        with open('results.json', 'a') as f:
            accuracy = {
                f"{mode}_accuracy": results.accuracy
            }
            dump_data = config | accuracy
            json.dump(dump_data, f, indent=2)


if __name__ == '__main__':
    print(f"Loaded base config from {defaultconfig_path}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=["auto", "cpu", "jit", "gpu"])
    # parser.add_argument('--config', default=wd / 'configs/miniseer/default_miniseer_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/microseer/default_microseer_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/citeseer/default_citeseer_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/cora/default_cora_config.yaml')
    parser.add_argument('--config', default=wd / 'configs/pubmed/default_pubmed_config.yaml')
    args = parser.parse_args()
    main(args)
