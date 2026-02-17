import os
import time
import json
import argparse
import tempfile
import warnings
import xyaml as yaml
import pickle as pkl
import pathlib as pl
import multiprocessing
from itertools import product
from dataclasses import dataclass, field

import tqdm
import numpy as np
import superneuromat as snm
from superneuromat.util import getenvbool
# import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

from graph_data import GraphData
from utils import bidict

# typing:
from typing import Sequence, Any, TYPE_CHECKING, overload, IO
from superneuromat import Neuron, Synapse
from graph_data import Pname


wd = pl.Path(__file__).parent.absolute()  # get the Path() of this python file

defaultconfig_path = wd / 'configs/default_config.yaml'

# this file is used to set the base config
# the per-dataset config is merged into this later
default_config = yaml.load(open(defaultconfig_path, 'r'))

do_print = False
ENV_PICKLE = 'SGNN_USE_PICKLE'


def unpack_single(func):
    def wrapper(x):
        return func(*x)
    return wrapper


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


def test_paper_from_pickle(paper_id: Pname, temp: str | os.PathLike):
    # For Windows & MacOS, the spawned process needs to run the whole file from scratch.
    # To skip the model loading overhead, this function loads a saved copy of the model from a pickle file.
    with open(temp, 'rb') as f:
        d = pkl.load(f)
    return test_paper(paper_id, d)


def test_paper_from_pickle1(x):
    # multiprocessing needs a top-level function that only takes one argument.
    paper_id, temp = x
    return test_paper_from_pickle(paper_id, temp)


def test_paper1(x):
    # multiprocessing needs a top-level function that only takes one argument.
    paper_id, temp = x
    return test_paper(paper_id, temp)


def test_paper(paper_id: Pname, graph: SGNN):
    """Infer the topic of a single paper.
    Returns a tuple of (the ground-truth topic, a list of [ties]), and the spike count."""
    snn = graph.snn
    paper = graph.papers[paper_id]
    paper_neuron: Neuron = graph.paper_neurons[paper_id]

    snn.add_spike(0, paper_neuron, 100.0)
    snn.simulate(time_steps=graph.config["simtime"])

    # Analyze the weights between the test paper neuron and topic neurons
    topic_weights = []
    for topic_id, topic_paper_id in graph.topic_neurons.items():
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
                feature_to_topic = snn.get_synapse(feature_neuron, graph.topic_neurons[topic])
                score += paper_to_feature.weight * feature_to_topic.weight
            topic_scores.append((topic, score))
        # choose papers with the highest score
        _best_topic, max_score = max(topic_scores, key=lambda x: x[1])
        ties = [topic for topic, score in topic_scores if score == max_score]

    if (r := graph.resolution_order):
        # reorder ties by resolution order
        ties = [k for k in r if k in ties]

    actual_topic = paper.label
    total_spikes = snn.ispikes.sum()
    snn.release_mem()
    del snn, graph
    # match = actual_topic == best_topic
    return (actual_topic, ties), total_spikes


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
    """Create an SGNN object based on the config file."""
    config = base_config.copy()
    with open(args.config, 'r') as f:
        config.update(yaml.load(f))  # this config overrides entries in the default config

    if do_print:
        print(f"Loaded dataset-specific config from {args.config}")
        print(f"This is the {config['dataset']} dataset over the {config['mode']} split.")

    graph = SGNN(name=config["dataset"], config=config)
    graph.load_all()
    graph.make_network()

    return graph


def evaluate(bundles, processes=1, temp=None):
    """Evaluate all bundles to infer the topic of each paper.

    Parallelizes the evaluation using the `process_map` function from `tqdm.contrib.concurrent`.
    For each paper, in a separate thread, we load the model from the temporary file, and then run the simulation.

    Args:
        bundles (list[tuple[str | int, str | SGNN]]): List of tuples containing the paper ID
            and the either the temporary file path or the SGNN object.
        processes (int, optional): Number of processes to use for parallel evaluation. Defaults to 1.
        temp (str, optional): Path to the temporary file. Defaults to None. Pass this if you want to evaluate
            the model from the pickle file.
    """
    eval_time = time.time()
    if processes == 1:
        func = test_paper if temp is None else test_paper_from_pickle
        # single-process evaluation
        x = [func(*bundle) for bundle in tqdm.tqdm(bundles)]
    else:
        func = test_paper1 if temp is None else test_paper_from_pickle1
        try:  # multi-process evaluation
            x = process_map(func, bundles, max_workers=processes, chunksize=1)  # with tqdm
            # pool = Pool(2)
            # x = pool.map(test_paper, bundles)
            pass
        finally:  # clean up and delete the temp file no matter what
            temp.close()  # close the temp file
            os.unlink(temp.name)  # DON'T COMMENT OUT THIS.
    eval_time = time.time() - eval_time
    if do_print:
        print(f"Evaluation time: {eval_time} seconds")
    return x


if TYPE_CHECKING:
    @overload
    def evaluate(bundles: Sequence[tuple[Pname, str | os.PathLike]],
                 processes=1, temp: IO = ...): ...  # type: ignore[override]
    @overload
    def evaluate(bundles: Sequence[tuple[Pname, SGNN]],
                 processes=1, temp: None = None): ...


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
    legacy: int | None = None

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
            f"Legacy:    {self.legacy} / {self.n} ({self.legacy / self.n * 100:.2f}%)" if self.legacy is not None else '',
            f"Total spikes: {np.sum(self.spikes)}",
        ])


def calculate_accuracy(results, resolution_order, name=''):
    n = len(results)
    results, spikes = zip(*results)
    # accuracy = np.sum(results) / n
    tp, tn, fp, fn = score(results, resolution_order)
    correct = np.sum([ans in guesses for ans, guesses in results])
    perfect = np.sum([set([ans]) == set(guesses) for ans, guesses in results])
    legacy = np.sum([guesses[0] == ans for ans, guesses in results])
    return Results(n=n, tp=tp, tn=tn, fp=fp, fn=fn, correct=correct, perfect=perfect, spikes=spikes, name=name, legacy=legacy)


def make_bundles(graph):
    """Build a list of (paper_id, graph) tuples of papers to evaluate.
    Each tuple contains arguments to pass to the ``test_paper(paper_id, graph)`` function."""
    return [(paper_id, graph) for paper_id in graph.selected_papers]


def make_temp_bundles(graph):
    """save model to file for multiprocessing.

    This makes evaluation faster on MacOS and Windows due to the spawn multiprocessing method.
    https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

    On such platforms, the `fork` method is not available, so this entire file is run for each process.
    Saving the model to a file allows us to simply load the file, instead of having to
    re-create the entire model each time.

    Returns  a list of (paper_id, graph) tuples of papers to evaluate.
    Each tuple contains arguments to pass to the ``test_paper(paper_id, graph)`` function."
    """
    temp = tempfile.NamedTemporaryFile(delete=False)
    with open(temp.name, 'wb') as f:
        pkl.dump(graph, f)
    # create a list of (paper_id, pickled_file) tuples for multiprocessing
    return temp, [(paper_id, temp.name) for paper_id in graph.selected_papers]


def main(args):

    model_time = time.time()

    graph = make_graph(args)
    model_time = time.time() - model_time
    if do_print:
        print(f"Time to load dataset and create model: {model_time} seconds")

    config = graph.config
    processes = graph.mp_processes(args.backend)
    mode = config['mode']
    if config['test_only_first']:
        graph.selected_papers = graph.selected_papers[:config['test_only_first']]

    # if there's a tie among topics, choose the topic closest to [0]
    # resolution_order = sorted(graph.topics, key=graph.topic_prevalence().get, reverse=True)  # sort topics by prevalence
    # resolution_order = reversed(list(graph.topic_neurons))  # sort topics by reverse load order
    resolution_order = list(graph.topic_neurons)  # sort topics by load order
    graph.resolution_order = resolution_order

    # with open('model.json', 'w') as f:
    #     graph.snn.saveas_json(f, array_representation="base85")

    if do_print:
        print(graph.snn.pretty(10))
        print("Topic breakdowns:")
        print(graph.topic_breakdowns())

    use_pickle = (
        x if (x := args.pickle) is not None else
        x if (x := getenvbool(ENV_PICKLE, default=None)) is not None else
        multiprocessing.get_start_method() == 'spawn'
    )

    if use_pickle:
        temp, bundles = make_temp_bundles(graph)
        del graph  # unload data and SNN to save memory

        # test loading the model from pickle
        load_time = time.time()
        with open(temp.name, 'rb') as f:
            d = pkl.load(f)
        del d
        load_time = time.time() - load_time
        if do_print:
            print(f"Time to load model from pickle: {load_time} seconds")

        x = evaluate(bundles, processes, temp)
    else:
        x = evaluate(make_bundles(graph), processes)

    results = calculate_accuracy(x, resolution_order, mode.title())
    if do_print:
        print(results)

    if config.get("dump_json", None):
        with open('results.json', 'a') as f:
            accuracy = {
                f"{mode}_accuracy": results.accuracy
            }
            dump_data = config | accuracy
            json.dump(dump_data, f, indent=2)


def main_parametrized(
    override_config_path=None,
    override_config=None,
    backend=None,  # None: use config backend or auto-select in SNM
    callback=None,
    pickle=None,
    base_config=default_config
):
    """Run the main function with the given arguments.

    Args:
        override_config_path (str | None, optional): Path to a config file to override the default config. Defaults to None.
        override_config (dict | None, optional):
            Dictionary of additional config values. Entries will override those in ``override_config_path``. Defaults to None.
        backend (str | None, optional): Backend to use for the SNN.
            Defaults to None meaning use config value, or if that is also None or 'auto', asks SNM to auto-select the backend.
        callback (callable | None, optional): Callback function to modify the graph or resolution order. Defaults to None.
        pickle (str | None, optional): Force using (True) or not using (False) a pickle file for each evaluation.
            Defaults to None, which means to either take the value of SGNN_USE_PICKLE environment variable if set.
            Otherwise, the default is to use a pickle file to speed up evaluation on MacOS and Windows.
        base_config (dict, optional): Base config to use. Defaults to ``default_config``.
    """
    config = base_config.copy()
    if override_config_path:
        with open(override_config_path, 'r') as f:
            config.update(yaml.load(f))  # this config overrides entries in the default config
    if override_config:
        config.update(override_config)

    graph = SGNN(name=config["dataset"], config=config)
    graph.load_all()
    graph.make_network()
    processes = graph.mp_processes(backend)
    mode = config['mode']
    if config['test_only_first']:
        graph.selected_papers = graph.selected_papers[:config['test_only_first']]

    # if there's a tie among topics, choose by whichever was loaded first
    order = graph.resolution_order = list(graph.topic_neurons)

    if callable(callback):
        # if callback returns a new order, use that
        order = new if (new := callback(graph)) is not None else order

    if (
        pickle if pickle is not None else
        x if (x := getenvbool(ENV_PICKLE, default=None)) is not None else
        multiprocessing.get_start_method() == 'spawn'
    ):
        temp, bundles = make_temp_bundles(graph)
        del graph  # unload data and SNN to save memory
        x = evaluate(bundles, processes, temp)
    else:
        x = evaluate(make_bundles(graph), processes)

    return calculate_accuracy(x, order, mode.title())


if __name__ == '__main__':
    do_print = True
    print(f"Loaded base config from {defaultconfig_path}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=["auto", "cpu", "jit", "gpu"])
    # parser.add_argument('--config', default=wd / 'configs/miniseer/default_miniseer_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/microseer/default_microseer_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/citeseer/default_citeseer_config.yaml')
    parser.add_argument('--config', default=wd / 'configs/cora/default_cora_config.yaml')
    # parser.add_argument('--config', default=wd / 'configs/pubmed/default_pubmed_config.yaml')
    parser.add_argument('--pickle', action='store_true', default=None)
    parser.add_argument('--nopickle', action='store_false', dest='pickle', default=None)
    args = parser.parse_args()
    main(args)
