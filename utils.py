import argparse
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, classification_report
import sys
from tabulate import tabulate
from typing import List, Tuple, Dict, Any


def plot_confusion_matrix(cm, labels):
    """
    Plot the confusion matrix 'cm' with the given 'labels'.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def evaluate_predictions(graph, predictions):
    """
    Build true_labels and pred_labels from 'predictions', then print confusion matrix and classification report.
    """
    true_labels = []
    pred_labels = []
    
    for (paper, predicted_topic, _) in predictions:
        actual_topic = graph.paper_to_topic[paper]
        true_labels.append(actual_topic)
        pred_labels.append(predicted_topic)
    
    # Unique set of topics encountered
    all_topics = list(set(true_labels + pred_labels))
    
    cm = confusion_matrix(true_labels, pred_labels, labels=all_topics)
    print("Confusion Matrix:\n", cm)
    
    report = classification_report(true_labels, pred_labels, labels=all_topics, target_names=all_topics)
    print("\nClassification Report:\n", report)
    
    # Optionally, plot confusion matrix:
    plot_confusion_matrix(cm, all_topics)


def print_paper_spikes(model, paper_neurons):
    """
    Print which paper neurons spiked at each time step, 
    assuming model.spike_train[t] is a binary array 
    indicating spike (1) or no spike (0) for each neuron.
    """
    for t, spike_array in enumerate(model.spike_train):
        # 'spike_array' is presumably something like [0, 1, 0, 0, 1, ...]
        spiking_papers = []
        for paper_id, neuron_id in paper_neurons.items():
            # Check if the value at index 'neuron_id' is 1 (i.e., a spike)
            if spike_array[neuron_id] == 1:
                spiking_papers.append(neuron_id)

        if spiking_papers:
            print(f"Time {t}: Paper Neurons that spiked: {spiking_papers}")


def plot_confusion_matrix(cm, labels):
    """
    Plot the confusion matrix 'cm' with the given 'labels'.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    
    # Print numeric values in the cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
        List[str]: List of unknown arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the input config file",
        default="./configs/citeseer/default_citeseer_config.yaml"
    )
    parser.add_argument(
        "--test-bo",
        action="store_true",
        help="Run the BO in testing mode"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        required=False,
        help="Name of the project",
        default="gnn-bo"
    )   
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def print_introduction(args: argparse.Namespace, config: omegaconf.DictConfig, kwargs: Dict[str, Any], delimiter: str = "*", delimiter_width: int = 80) -> None:
    """
    Print an introduction with the default config and overridden values.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        config (omegaconf.DictConfig): Baseline configuration.
        kwargs (Dict[str, Any]): Dictionary of overridden keyword arguments.
        delimiter (str, optional): Delimiter character for the printed output. Defaults to "*".
        delimiter_width (int, optional): Width of the delimiter line. Defaults to 80.
    Returns:
        None
    
    """
    print(delimiter * delimiter_width)
    print(f"\nConfig file: {args.config}\n")
    print("Default Config:")
    default_config_table = []
    # Print each level of the configuration
    for key, value in config.items():
        if isinstance(value, DictConfig):
            for subkey, subvalue in value.items():
                default_config_table.append([f"{key}.{subkey}", subvalue])
        else:
            default_config_table.append([key, value])
    print(tabulate(default_config_table, headers=["Key", "Value"], tablefmt="grid"))

    if kwargs:
        print("\nOverridden Config:")

        overridden_config_table = []

        for key, value in kwargs.items():
            if isinstance(value, DictConfig):
                for subkey, subvalue in value.items():
                    overridden_config_table.append([f"{key}.{subkey}", config[key][subkey], subvalue])
            else:
                overridden_config_table.append([key, config[key], value])

        # overridden_values_table = [[key, config[key], value] for key, value in kwargs.items()]
        print(tabulate(overridden_config_table, headers=["Key", "Original Value", "New Value"], tablefmt="grid"))
    
    print()
    print(delimiter * delimiter_width)


def validate_overrides(loaded_config: DictConfig, argued_config: DictConfig) -> DictConfig:
    """
    Parse the list of keyword arguments into a dictionary.

    Args:
        kwargs_list (List[str]): List of keyword arguments in the form of key=value.

    Returns:
        Dict[str, Any]: Dictionary of parsed keyword arguments.
    """
    # Find unknown keys in the CLI configuration
    unknown_keys = [key for key in argued_config if key not in loaded_config]

    if unknown_keys:
        print(f"Error: Unknown keys in command-line arguments: {unknown_keys}")
        sys.exit(1)  # Exit with an error code

    # Merge the CLI configuration into the base configuration
    merged_cfg = OmegaConf.merge(loaded_config, argued_config)
    return merged_cfg