import os
import warnings
import pathlib as pl
from collections import Counter
from dataclasses import dataclass, field
from os.path import expandvars

import numpy as np
import networkx as nx
import superneuromat as snm
from sklearn.preprocessing import MultiLabelBinarizer

from typing import Iterable

Pname = str | int

import torch
import numpy as np

# --- CRITICAL FIX for PyTorch >= 2.6 + OGB MAG240M ---
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # Force legacy behavior so OGB split_dict.pt can load
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

from ogb.lsc import MAG240MDataset
DATA_ROOT="/lustre/orion/lrn088/scratch/srk20/data/"
# -----------------------------------------------------



@dataclass
class Paper:
    """Data structure for a single paper."""
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
    """Loads and facilitates access to Citation Network data.

    Handles loading of papers, topics, and features.
    Also provides methods for selecting papers for training, validation, and test sets.

    The ``config`` defines how the data should be loaded and split, and where the data is located.
    See the ``configs/default_config.yaml`` file for an example.

    The data is not loaded at instantiation.
    Use the ``load_all()`` method to load all data and perform splits.
    """
    def __init__(self, name, config, **kwargs):
        self.name = name
        config.update(kwargs)
        self.config = config
        self.ordered_paper_ids: list[Pname] = []
        self.papers: dict[Pname, Paper] = {}
        self.topics: list[str] = []  # the list of topics
        self.resolution_order: Iterable[Pname] = []
        self._edges_path = None
        self._nodes_path = None
        self.data_root: pl.Path | None = pl.Path(expandvars(p)) if (p := config.get("data_root", None)) else None
        self.edges_path = expandvars(self.config["edges_path"])
        self.nodes_path = expandvars(self.config["nodes_path"])
        self.nodes_feature_path = expandvars(self.config["nodes_features_path"])
        self.train_papers: list[Pname] = []
        self.validation_papers: list[Pname] = []
        self.test_papers: list[Pname] = []
        self.mlb = MultiLabelBinarizer()

        self.seed = config.get("seed", None)

    def resolve_path_with_root(self, path: os.PathLike, root: os.PathLike | None) -> pl.Path:
        if not isinstance(path, pl.Path):
            path = pl.Path(path)
        if not isinstance(root, pl.Path) and root is not None:
            root = pl.Path(root)
        if path.expanduser().is_absolute() and root is not None:
            warnings.warn(f"WARNING: Path {path} is absolute, but data_root is set to {root}. "
                          "Ignoring data_root.", RuntimeWarning, stacklevel=3)
            return path.expanduser().resolve()
        return (root / path).expanduser().resolve()

    @property
    def edges_path(self):
        assert self._edges_path is not None
        return self.resolve_path_with_root(self._edges_path, self.data_root)

    @edges_path.setter
    def edges_path(self, path: os.PathLike):
        self._edges_path = path

    @property
    def nodes_path(self):
        assert self._nodes_path is not None
        return self.resolve_path_with_root(self._nodes_path, self.data_root)

    @nodes_path.setter
    def nodes_path(self, path: os.PathLike):
        self._nodes_path = path

    def remove_missing(self):
        missing = [paper.idx for paper in self.papers.values() if not paper.label]
        for paper in missing:
            del self.papers[paper]

    def load_all(self, remove_missing=True):
        if self.config["dataset"]=="mag240m":
            self.dataset = MAG240MDataset(root=DATA_ROOT)

        self.load_graph()
        self.load_papers()
        self.load_topics()
        print("Papers, topics and connectivity added in GraphData object.")
        # some papers appear as a citation but don't have a label. Remove those papers.
        if remove_missing:
            self.remove_missing()
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
        self.ordered_paper_ids = order = []
        if self.nodes_path.suffix == ".content":
            with open(self.nodes_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                paper_id, *_features, label = line.strip().split()
                topics.add(label)
                self.papers[paper_id].label = label
                order.append(paper_id)
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
                order.append(paper_idx)
        #########################################################
        #### Case for MAG240M dataset:
        #elif self.nodes_path.suffix == ".npy":
        elif self.config["dataset"] == "mag240m":
            topics = np.arange(self.dataset.num_classes)
            print("MAG240M topics:", topics)
        #########################################################
        self.topics = list(topics)
        self.mlb = MultiLabelBinarizer(classes=self.topics)
        # force load order
        self.papers = {k: self.papers[k] for k in order}

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
         #########################################################
        #### Case for MAG240M dataset:
        elif self.nodes_feature_path.suffix == ".npy":
            sel.paper[paper_id].features = np.load(self.nodes_feature_path) # dataset.num_paper_features, /lustre/orion/lrn088/scratch/srk20/data/mag240m_kddcup2021/processed/paper/node_feat.npy
        #########################################################

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
        ##################################
        ##### FOR MAG20M:
        elif self.edges_path.suffix == ".npy":
            #print("Location of edges:",self.edges_path)
            #self.graph = nx.from_edgelist(np.load(self.edges_path))
            edge_index_cites = self.dataset.edge_index('paper', 'paper')
            #edge_list = list(zip(edge_index_cites[0], edge_index_cites[1]))
            #self.graph = nx.Graph(edge_list)
            self.graph = edge_index_cites   # np.ndarray
        #####################################

    def load_papers(self):
        if self.config["dataset"]=="mag240m":
            self.papers = {}
            for k in range(self.dataset.num_papers):
                self.papers[k]=Paper(k)
            return

        for paper in self.graph.nodes:
            self.papers[paper] = Paper(paper)

    def train_val_test_split_legacy(self):
        if self.config["dataset"]=="mag240m":
            split_dict = self.dataset.get_idx_split()
            self.train_papers = split_dict['train'] # numpy array storing indices of training paper nodes
            self.validation_papers = split_dict['valid'] # numpy array storing indices of validation paper nodes
            self.test_papers = split_dict['test-dev'] # numpy array storing indices of test-dev paper nodes
            self.testchallenge_papers = split_dict['test-challenge'] # numpy array storing indices of test-challenge paper nodes

            return

        if (pick_evenly := self.config.get('pick_evenly', None)) is not None:
            pick_evenly = set(pick_evenly) - {'null', None}
            if pick_evenly != {'train', 'validation'}:
                warnings.warn("WARNING: You're using config['legacy_split'] which only supports "
                              "config['pick_evenly'] = ['train', 'validation']. "
                              "We'll ignore your 'pick_evenly' setting.", RuntimeWarning, 2)

        rng = np.random.RandomState(self.seed)
        paperlist = list(self.papers.values())

        def select_papers(topics_mult=20, used: set[int | str] | None = None):
            used = used or set()  # papers that should not be selected
            selected = []
            topic_counts = Counter()
            attempts = 0

            while len(selected) < len(self.topics) * topics_mult:
                idx = rng.randint(len(self.papers))
                paper = paperlist[idx]
                if (
                    topic_counts[paper.label] < topics_mult
                    and paper not in selected
                    and paper.idx not in used
                ):
                    selected.append(paper.idx)
                    used.add(paper.idx)
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

        def select_papers(pool, n=1, removefrom=()):
            """take & return the first `n` papers from the pool without replacement"""
            # bisect
            chosen, pool[:] = pool[0:n], pool[n:]
            # remove chosen papers from any auxiliary pools
            chosenset = set(chosen)
            for aux_pool in removefrom:
                aux_pool[:] = [paper for paper in aux_pool if paper not in chosenset]
            return chosen

        def select_topics_evenly(pool, n):
            """take `n` papers from the pool, evenly distributing them across topics as the stacks run out"""
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

        class Selector:  # chooses selection function based on if 'train'/'test'/'validation' is in config['pick_evenly']
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
        """Count the number of papers in each topic."""
        if papers is None:
            papers = self.papers.values()
        topic_counts = {k: 0 for k in self.topics}  # create a dictionary with all topics as keys
        for paper in papers:
            label = paper.label if isinstance(paper, Paper) else self.papers[paper].label
            topic_counts[label] += 1
        return topic_counts

    def topic_breakdowns(self):
        """Create a pandas DataFrame with the number of papers in each topic for all/test/train/validation sets."""
        import pandas as pd
        all_papers = self.topic_prevalence()
        test = self.topic_prevalence(self.test_papers)
        train = self.topic_prevalence(self.train_papers)
        validation = self.topic_prevalence(self.validation_papers)
        df = pd.DataFrame([all_papers, test, train, validation], index=['all', 'test', 'train', 'validation'])
        return df
