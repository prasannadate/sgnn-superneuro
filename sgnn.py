import numpy as np

from superneuromat.neuromorphicmodel import NeuromorphicModel


def load_network(graph, config):

    # Initialize the Neuromorphic Model
    model = NeuromorphicModel()

    # Dictionaries to hold neuron IDs
    topic_neurons = {}
    paper_neurons = {}
    feature_neurons = {}

    # Create paper neurons
    for node in graph.graph.nodes:
        if (node not in graph.paper_to_topic.keys()):
            continue
        neuron_id = model.create_neuron(
            threshold=config.paper.threshold,
            leak=config.paper.leak,
            reset_state=0.0,
            refractory_period=0,  # Will set below based on paper type
        )
        paper_neurons[node] = neuron_id

        # Set refractory periods based on paper type
        if node in graph.train_papers:
            model.neuron_refractory_periods[neuron_id] = config.train.ref
        elif node in graph.validation_papers:
            model.neuron_refractory_periods[neuron_id] = config.validation.ref
        elif node in graph.test_papers:
            model.neuron_refractory_periods[neuron_id] = config.test.ref

    # Create topic neurons
    for t in graph.topics:
        neuron_id = model.create_neuron(
            threshold=config.topic.threshold,
            leak=config.topic.leak,
            reset_state=0.0,
            refractory_period=0, 
        )
        topic_neurons[t] = neuron_id
        # Setting tau_minus is not directly supported

    # Connect paper neurons based on the graph edges
    for edge in graph.graph.edges:
        if (edge[0] not in graph.paper_to_topic.keys() or edge[1] not in graph.paper_to_topic.keys()):
            continue
        pre_id = paper_neurons[edge[0]]
        post_id = paper_neurons[edge[1]]
        variation_scale = 0.01

        w = config.graph.weight
        w *= (1.0 + np.random.normal(0, variation_scale))
        d = int(config.graph.delay)
        model.create_synapse(pre_id, post_id, weight=w, delay=d, enable_stdp=False)
        model.create_synapse(post_id, pre_id, weight=w, delay=d, enable_stdp=False)

    # Connect training papers to topics
    for paper in graph.train_papers:
        paper_id = paper_neurons[paper]
        topic_id = topic_neurons[graph.paper_to_topic[paper]]
        variation_scale = 0.01 

        w = config.train_to_topic.weight
        w *= (1.0 + np.random.normal(0, variation_scale))
        d = int(config.train_to_topic.delay)
        model.create_synapse(paper_id, topic_id, weight=w, delay=d, enable_stdp=False)
        model.create_synapse(topic_id, paper_id, weight=w, delay=d, enable_stdp=False)

    # Connect validation papers to topics with STDP
    for paper in graph.validation_papers:
        paper_id = paper_neurons[paper]
        for topic in graph.topics:
            topic_id = topic_neurons[topic]
            variation_scale = 0.01 

            w = config.validation_to_topic.weight
            w *= (1.0 + np.random.normal(0, variation_scale))
            d = int(config.validation_to_topic.delay)
            stdp_on = True
            model.create_synapse(
                paper_id, topic_id, weight=w, delay=d, enable_stdp=stdp_on
            )
            model.create_synapse(
                topic_id, paper_id, weight=w, delay=d, enable_stdp=stdp_on
            )

    # Connect test papers to topics with STDP
    for paper in graph.test_papers:
        paper_id = paper_neurons[paper]
        for topic in graph.topics:
            topic_id = topic_neurons[topic]
            variation_scale = 0.01 

            w = config.test_to_topic.weight
            w *= (1.0 + np.random.normal(0, variation_scale))
            d = int(config.test_to_topic.delay)
            stdp_on = True
            model.create_synapse(
                paper_id, topic_id, weight=w, delay=d, enable_stdp=stdp_on
            )
            model.create_synapse(
                topic_id, paper_id, weight=w, delay=d, enable_stdp=stdp_on
            )

    return model, paper_neurons, topic_neurons, feature_neurons


def test_paper(x):
    paper = x[0]    # paper ID of the paper you want to test
    graph = x[1]    # loaded graph
    config = x[2]   # config parameters

    # Load the network
    model, paper_neurons, topic_neurons, feature_neurons = load_network(graph, config)

    # Add a spike to the test paper neuron at time step 1 with a value sufficient to trigger a spike
    test_paper_id = paper_neurons[paper]
    model.add_spike(0, test_paper_id, value=config.input_spike_value)

    correct_topic_label = graph.paper_to_topic[paper]  # e.g. "Neural_Nets"
    correct_topic_id = topic_neurons[correct_topic_label]
    #model.add_spike(1, correct_topic_id, value=100.0)

    # Determine the number of time steps for STDP
    simtime = int(config.simtime) 
    time_steps = int(config.get("stdp_time_steps", simtime))

    # Set up STDP parameters if not already set
    if not model.stdp:
        apos_value = config.stdp.A_pos
        aneg_value = config.stdp.A_neg

        # Create lists of Apos and Aneg values with length equal to time_steps
        Apos = [apos_value] * time_steps
        Aneg = [aneg_value] * time_steps
        model.stdp_setup(
            time_steps=time_steps,
            Apos=Apos,
            Aneg=Aneg,
            positive_update=True,
            negative_update=True,
        )

    # Prepare the model for simulation
    model.setup()

    # Simulate the model
    model.simulate(time_steps=simtime)
    #print("Printing", test_paper_id, correct_topic_id)
    #print_paper_spikes(model, topic_neurons)

    
    # Analyze the weights between the test paper neuron and topic neurons
    #min_weight = float('inf')
    min_weight = -100
    min_topic = None
    predicted_topic = None
    for topic, topic_id in topic_neurons.items():
        # Find the synapse from topic neuron to test paper neuron
        synapse_indices = [
            i for i, (pre, post) in enumerate(zip(model.pre_synaptic_neuron_ids, model.post_synaptic_neuron_ids))
            #if pre == topic_id and post == test_paper_id
            if pre == test_paper_id and post == topic_id
        ]
        if synapse_indices:
            idx = synapse_indices[0]
            weight = model.synaptic_weights[idx]
            print(f"Topic: {topic}, Paper: {paper}, Weight: {weight}")
            if weight > min_weight:
                min_weight = weight
                min_topic = topic
                predicted_topic = topic

    # Determine if the predicted topic matches the actual topic
    actual_topic = graph.paper_to_topic[paper]
    retval = 1 if actual_topic == min_topic else 0
    if retval == 1:
        print(f"MIN VAL for {paper} Topic {min_topic} CORRECT")
    else:
        print(f"MIN VAL for {paper} Topic {min_topic} WRONG, Expected {actual_topic}")

    # Optionally, plot the weights
    if config.get("monitors", False):
        weights = []
        topics = []
        for topic, topic_id in topic_neurons.items():
            synapse_indices = [
                i for i, (pre, post) in enumerate(zip(model.pre_synaptic_neuron_ids, model.post_synaptic_neuron_ids))
                if pre == topic_id and post == test_paper_id
            ]
            if synapse_indices:
                idx = synapse_indices[0]
                weights.append(model.synaptic_weights[idx])
                topics.append(topic)
        plt.bar(topics, weights)
        plt.xlabel('Topics')
        plt.ylabel('Weights')
        plt.title('Weights from Topic Neurons to Test Paper Neuron')
        plt.show()

    #return retval
    return (paper, predicted_topic, retval)


def evaluate_network(graph, test_papers, get_predicted_topic_for_paper):
    true_labels = []
    pred_labels = []

    for paper in test_papers:
        true_label = graph.paper_to_topic[paper]
        predicted_label = get_predicted_topic_for_paper(paper)  # your code here
        true_labels.append(true_label)
        pred_labels.append(predicted_label)

    # Unique topics used
    topics_list = list(set(true_labels + pred_labels))

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=topics_list)
    print("Confusion Matrix:")
    print(cm)

    # Classification report
    report = classification_report(true_labels, pred_labels, labels=topics_list, target_names=topics_list)
    print("\nClassification Report:")
    print(report)

    # Optional: plot confusion matrix
    plot_confusion_matrix(cm, topics_list)
