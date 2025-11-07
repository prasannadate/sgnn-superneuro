#!/usr/bin/env python3
""" Contains:
    load_network()
    test_paper()
"""
import json
import time
from multiprocessing import Pool

import networkx as nx
import numpy as np
import superneuromat as snm
import yaml

from graph_data import GraphData


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
        if pre != post:
            model.create_synapse(pre, post, weight=config["graph_weight"], delay=config["graph_delay"], stdp_enabled=False)
            model.create_synapse(post, pre, weight=config["graph_weight"], delay=config["graph_delay"], stdp_enabled=False)

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
    paper_id = paper_neurons[paper]
    model.add_spike(0, paper_id, 100.0)
    timesteps = config["simtime"]
    model.stdp_setup(time_steps=config["stdp_timesteps"],
        Apos=config["apos"], Aneg=config["aneg"] * config["stdp_timesteps"], negative_update=True, positive_update=True)
    model.setup(sparse=True)
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
    config = yaml.safe_load(open("configs/cora/apos_cora_config.yaml"))

    np.random.seed(config["seed"])
    graph = GraphData(config["dataset"],
                      seed=config['seed'],
                      data_dir='./data')

    i = 0
    correct = 0
    total = 0
    num_spikes = 0

    start_time = time.time()

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

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")