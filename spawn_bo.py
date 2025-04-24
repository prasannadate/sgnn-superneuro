# standard pip imports
import copy
from multiprocessing import Pool
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf
from skopt import Space
from skopt.space import Categorical, Integer, Real
import sys
import wandb

# project specific pip imports
from lmao.solver import BOSolver
from superneuromat.neuromorphicmodel import NeuromorphicModel

# local imports
from graph_data import GraphData
from sgnn import (
    test_paper,
)
from utils import (
    evaluate_predictions,
    parse_args,
    plot_confusion_matrix,
    print_introduction,
    print_paper_spikes,
    validate_overrides,
)

def solve_gnn(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10) -> float:
    """
    Solve the GNN model with the specific BO configuration.
    """
    local_config = copy.deepcopy(DEFAULT_CONFIG)

    override_config = DictConfig({
        "graph": {
            "weight": float(arg1)
        },
        "paper": {
            "leak": float(arg2),
            "threshold": float(arg3)
        },
        "stdp": {
            "A_pos": float(arg4),
            "A_neg": float(arg5)
        },
        "test_to_topic": {
            "weight": float(arg6)
        },
        "topic": {
            "leak": float(arg7),
            "threshold": float(arg8)
        },
        "train_to_topic": {
            "weight": float(arg9)
        },
        "validation_to_topic": {
            "weight": float(arg10)
        },
    })

    local_config = omegaconf.OmegaConf.merge(local_config, override_config)

    print("Running with config:")
    print(local_config)

    wandb_run = wandb.init(
        project="testing",
        config=OmegaConf.to_container(local_config),
    )

    np.random.seed(local_config.seed)

     # Spike each of the test papers in simulation
    i = 0
    correct = 0
    total = 0
    papers = []
    # accuracy = 1 * np.random.rand()

    graph = GraphData(local_config.dataset, local_config)    
    
    if (config["monitors"] == True):
        for i in range(0, len(graph.validation_papers), 20):
            paper = graph.validation_papers[i]
            x = [paper, graph, local_config]
            test_paper(x)
        sys.exit()

    with Pool(local_config.processes) as pool:
        if local_config.mode == "validation":
            iterator = graph.validation_papers
        elif local_config.mode == "test":
            iterator = graph.test_papers
        
        for paper in iterator:
            papers.append([paper, graph, local_config])

        # predictions will be a list of tuples: (paper, predicted_topic, retval)
        predictions = pool.map(test_paper, papers)
        
        # You already print accuracy using retval
        accuracy = np.mean([ret for (_, _, ret) in predictions])

        if local_config.mode == "validation":
            print("Validation Accuracy:", accuracy)
        elif local_config.mode == "test":
            print("Testing Accuracy:", accuracy)

        # Evaluate with confusion matrix, classification report, etc.
        evaluate_predictions(graph, predictions)

    wandb.log({"accuracy": accuracy})
    wandb_run.finish()

    return np.array([accuracy])


def main(config: omegaconf.DictConfig) -> None:
    """
    Main function to execute the script.

    Args:
        config (omegaconf.DictConfig): Merged configuration.
    """

    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config

    search_space = Space([
        Categorical([1.0, 5.0, 10.0, 15.0, 20.0], name="graph_weight"),
        Categorical([1.0, 10.0, 100.0, 100000.0], name="paper_leak"),
        Categorical([0.25, 0.5, 0.75, 1.0, 1.5], name="paper_threshold"),
        Categorical([0.01, 0.1, 1.0, 10.0], name="stdp_A_pos"),
        Categorical([0.00001, 0.0001, 0.001, 0.01], name="stdp_A_neg"),
        Categorical([0.001, 0.005, 0.01, 0.05, 0.1], name="test_to_topic_weight"),
        Categorical([1.0, 10.0, 100.0, 100000.0], name="topic_leak"),
        Categorical([0.25, 0.5, 0.75, 1.0, 1.5], name="topic_threshold"),
        Categorical([0.5, 1.0, 10.0, 100.0], name="train_to_topic_weight"),
        Categorical([0.001, 0.005, 0.01, 0.05, 0.1], name="validation_to_topic_weight"),
    ])

    optimizer_config = DictConfig({
        "num_initial_points": config.bo.num_initial_points,
        "max_iterations": config.bo.max_iterations,
        "num_processes": 1,
        "num_repeats": 1,
        "optimizer_class": "gp-cpu",
        "optimizer": {
            "num_initial_points": config.bo.num_initial_points,
        },
        "seed": config.seed,
    })

    # solve_gnn({})

    solver = BOSolver(optimizer_config)
    result = solver.solve(
        solve_gnn,
        use_lp=False,
        search_space=search_space
    )


if __name__ == "__main__":
    args, unknown_args = parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    override_config = omegaconf.OmegaConf.from_dotlist(unknown_args)
    validate_overrides(config, override_config)
    print_introduction(args, config, override_config)
    config = omegaconf.OmegaConf.merge(config, override_config)
    main(config)
