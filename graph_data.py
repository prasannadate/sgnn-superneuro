#!/usr/bin/env python3
"""
Graph data loading and processing class
"""
from pathlib import Path
import networkx as nx
import numpy as np


class GraphData():
    def __init__(self, name, hyperparameters, data_dir):
        """

        :param name: benchmark dataset name
        :param hyperparameters: contains configuration parameters
        :param data_dir: directory where the data is stored
        """
        self.name = name
        self.hyperparameters = hyperparameters
        self.data_dir = Path(data_dir)
        self.paper_to_topic = {} # maps the paper ID in the dataset to its topic ID
        self.index_to_paper = []    # creates an index for each paper
        self.topics = []            # the list of topics 
        self.train_papers = []
        self.validation_papers = []
        self.test_papers = []
        self.load_topics()
        if self.name != "microseer":
            # FIXME why?
            self.train_val_test_split()
        else:
            self.train_val_test_split_small()
        self.load_features()
        self.load_graph()

    def load_topics(self):
        if (self.name == "cora"):
            with (self.data_dir / "Cora" / "group-edges.csv").open('r') as f:
                lines = f.readlines()

            for line in lines:
                fields = line.strip().split(",")
                if (fields[1] not in self.topics):
                    self.topics.append(fields[1])
                self.paper_to_topic[fields[0]] = fields[1]
                self.index_to_paper.append(fields[0])
        elif (self.name == "citeseer"):
            with (self.data_dir / "citeseer" / "citeseer.content").open('r') as f:
                lines = f.readlines()

            for line in lines:
                fields = line.strip().split()
                if (fields[-1] not in self.topics):
                    self.topics.append(fields[-1])
                print(fields[0])
                self.paper_to_topic[fields[0]] = fields[-1]
                self.index_to_paper.append(fields[0])   
        elif (self.name == "pubmed"):
            with (self.data_dir / "Pubmed-Diabetes" / "data" / "Pubmed-Diabetes.NODE.paper.tab").open('r') as f:
                lines = f.readlines()

            for line in lines[2:]: # skip header lines
                fields = line.strip().split()
                if (fields[1] not in self.topics):
                    self.topics.append(fields[1])
                self.paper_to_topic["paper:"+fields[0]] = fields[1]
                self.index_to_paper.append("paper:"+fields[0])

    def load_features(self):
        self.features = {} # keyed on paper ID, value is the feature vector
        if (self.name == "cora"):
            with (self.data_dir / "Cora" / "cora" / "cora.content").open('r') as  f:
                lines = f.readlines()

            for line in lines:
                fields = line.strip().split()
                paper_id = fields[0]
                feature = [int(x) for x in fields[1:-1]]
                self.features[paper_id] = feature
                self.num_features = len(feature)
        elif (self.name == "citeseer"):
            with (self.data_dir / "citeseer" / "citeseer.content").open('r') as  f:
                lines = f.readlines()

            for line in lines:
                fields = line.strip().split()
                paper_id = fields[0]
                feature = [int(x) for x in fields[1:-1]]
                self.features[paper_id] = feature
                self.num_features = len(feature)
            print(f"NUM PAPERS WITH FEATURES: {len(self.features.keys())}")
        elif (self.name == "pubmed"):
            with (self.data_dir / "Pubmed-Diabetes" / "data" / "Pubmed-Diabetes.NODE.paper.tab").open('r') as  f:
                lines = f.readlines()

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
            self.graph = nx.read_edgelist(str(self.data_dir / 'Cora' / 'edges.csv'),
                                          delimiter=",")
        elif (self.name == "citeseer"):
            self.graph = nx.read_edgelist(str(self.data_dir / 'citeseer' / 'citeseer.cites'))

        elif (self.name == "pubmed"):
            self.graph = nx.read_edgelist(str(self.data_dir / 'Pubmed-Diabetes' / 'data' / 'edge_list.csv'),
                                          delimiter=",")

    def train_val_test_split(self):
        np.random.seed(self.hyperparameters["seed"])
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
