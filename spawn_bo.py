# standard pip imports
import copy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf
import os
import pickle
from skopt import Optimizer, Space
from skopt.space import Categorical
import time

# project specific pip imports
from gnn_citation_networks import GraphData, test_paper
from utils import (
    parse_args,
    print_introduction,
    validate_overrides,
)

BO_TESTING_MODE = False
PROJECT_NAME = "GNN-BO"
DEFAULT_CONFIG = None

def solve_gnn(arg1, arg2, arg3, arg4,
              arg5, arg6, arg7, arg8,
              arg9, arg10, arg11, arg12,
              arg13) -> float:
    """
    Solve the GNN model with the specific BO configuration.
    """
    local_config = copy.deepcopy(DEFAULT_CONFIG)

    override_config = DictConfig({
        "graph_weight": float(arg1),
        "paper_leak": float(arg2),
        "paper_threshold": float(arg3),
        "topic_leak": float(arg4),
        "topic_threshold": float(arg5),
        "train_to_topic_weight": float(arg6),
        "validation_to_topic_weight": float(arg7),
        "test_to_topic_weight": float(arg7),
        "apos": [float(x) for x in [arg8, arg9, arg10, arg11, arg12]],
        "aneg": [float(arg13)],
        "mode": "validation"
    })

    # print("Override config:")
    # print(json.dumps(OmegaConf.to_container(override_config), indent=4))

    # Extract keys from both configs
    baseline_keys = set(OmegaConf.to_container(local_config, resolve=True).keys())
    override_keys = set(OmegaConf.to_container(override_config, resolve=True).keys())

    # Identify keys that are in override_config but not in local_config
    unexpected_keys = override_keys - baseline_keys

    if unexpected_keys:
        print(f"Unexpected keys in override_config that do not exist in local_config: {unexpected_keys}")
        raise ValueError(f"Unexpected keys in override_config that do not exist in local_config: {unexpected_keys}")

    local_config = omegaconf.OmegaConf.merge(local_config, override_config)

    local_config = OmegaConf.to_container(local_config)

    np.random.seed(local_config["seed"])

     # Spike each of the test papers in simulation
    papers = []

    if BO_TESTING_MODE:
        accuracy = np.random.rand()
    else:
        graph = GraphData(local_config["dataset"], local_config)    

        with Pool(local_config["processes"]) as pool:
            if local_config["mode"] == "validation":
                iterator = graph.validation_papers
            elif local_config["mode"] == "test":
                iterator = graph.test_papers
            
            for paper in iterator:
                papers.append([paper, graph, local_config])

            # predictions will be a list of tuples: (paper, predicted_topic, retval)
            predictions = pool.map(test_paper, papers)

            if local_config["mode"] == "validation":
                accuracy = sum(i[0] for i in predictions) / len(graph.validation_papers)
                print("Validation Accuracy:", accuracy)
            elif local_config["mode"] == "testing":
                accuracy = sum(i[0] for i in predictions) / len(graph.test_papers)
                print("Testing Accuracy:", accuracy)


    # wandb.log({"accuracy": accuracy})
    # wandb_run.finish()

    return accuracy


def main(config: omegaconf.DictConfig) -> None:
    """
    Main function to execute the script.

    Args:
        config (omegaconf.DictConfig): Merged configuration.
    """

    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config

    search_space = Space([
        Categorical([75.0, 100.0, 125.0], name="graph_weight"),
        Categorical([0.0, 1.0], name="paper_leak"),
        Categorical([0.5, 1.0, 1.5], name="paper_threshold"),
        Categorical([0.0, 1.0], name="topic_leak"),
        Categorical([0.5, 1.0, 1.5], name="topic_threshold"),
        Categorical([0.5, 1.0, 1.5], name="train_to_topic_weight"),
        Categorical([0.0005, 0.001, 0.005], name="validation_to_topic_weight"),
        Categorical([0.75, 1.0, 1.25], name="apos_0"),
        Categorical([0.25, 0.5, 0.75], name="apos_1"),
        Categorical([0.05, 0.1, 0.15], name="apos_2"),
        Categorical([0.005, 0.01, 0.015], name="apos_3"),
        Categorical([0.0005, 0.001, 0.0015], name="apos_4"),
        Categorical([0.00005, 0.0001, 0.00015], name="aneg_0"),
    ])

    np.random.seed(config.seed)

    num_iterations: int = config.bo.max_iterations
    num_initial_points: int = config.bo.num_initial_points

    optimizer = Optimizer(
        dimensions=search_space,
        base_estimator="GP",
        initial_point_generator="random",
        random_state=config.seed,
    )

    parameter_log = []
    accuracy_log = []
    ask_time_log = []
    eval_time_log = []
    tell_time_log = []

    os.makedirs(config.bo.output_dir, exist_ok=False)
    param_log_path: str = os.path.join(config.bo.output_dir, "parameter_log.npy")
    accuracy_log_path: str = os.path.join(config.bo.output_dir, "accuracy_log.npy")
    ask_time_log_path: str = os.path.join(config.bo.output_dir, "ask_time_log.npy")
    eval_time_log_path: str = os.path.join(config.bo.output_dir, "eval_time_log.npy")
    tell_time_log_path: str = os.path.join(config.bo.output_dir, "tell_time_log.npy")
    result_path: str = os.path.join(config.bo.output_dir, "result.pkl")


    for idx in range(num_initial_points + num_iterations):
        if idx <= num_initial_points:
            print(f"Initial point {idx + 1}/{num_initial_points}")
        else:
            print(f"Optimization iteration {idx + 1 - num_initial_points}/{num_iterations}")


        ask_time_start: float = time.time()
        next_x = optimizer.ask()
        ask_time_end: float = time.time()

        eval_time_start: float = time.time()
        f_val = solve_gnn(*next_x)
        eval_time_end: float = time.time()

        parameter_log.append(next_x[0])
        accuracy_log.append(f_val)

        tell_time_start: float = time.time()
        optimizer_result = optimizer.tell(next_x, -1 * f_val)
        tell_time_end: float = time.time()

        ask_time_log.append(ask_time_end - ask_time_start)
        eval_time_log.append(eval_time_end - eval_time_start)
        tell_time_log.append(tell_time_end - tell_time_start)


        # save the logs very iteration
        if idx % config.bo.save_iteration == 0:

            np.save(param_log_path, np.array(parameter_log))
            np.save(accuracy_log_path, np.array(accuracy_log))
            np.save(ask_time_log_path, np.array(ask_time_log))
            np.save(eval_time_log_path, np.array(eval_time_log))
            np.save(tell_time_log_path, np.array(tell_time_log))

            with open(result_path, "wb") as f:
                pickle.dump(optimizer_result, f)

            # plot the accuracy log sorted and unsorted
            plt.plot(accuracy_log)
            plt.title("Accuracy vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.savefig(os.path.join(config.bo.output_dir, "accuracy_plot.png"))
            plt.close()

            # plot the accuracy log sorted
            sorted_accuracy_log = np.array(accuracy_log)
            sorted_accuracy_log.sort()
            plt.plot(sorted_accuracy_log)
            plt.title("Sorted Accuracy vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.savefig(os.path.join(config.bo.output_dir, "sorted_accuracy_plot.png"))
            plt.close()

            # plot the ask time log
            plt.plot(ask_time_log)
            plt.title("Ask Time vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Time (s)")
            plt.savefig(os.path.join(config.bo.output_dir, "ask_time_plot.png"))
            plt.close()

            # plot the eval time log
            plt.plot(eval_time_log)
            plt.title("Eval Time vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Time (s)")
            plt.savefig(os.path.join(config.bo.output_dir, "eval_time_plot.png"))
            plt.close()

            # plot the tell time log
            plt.plot(tell_time_log)
            plt.title("Tell Time vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Time (s)")
            plt.savefig(os.path.join(config.bo.output_dir, "tell_time_plot.png"))
            plt.close()

if __name__ == "__main__":
    args, unknown_args = parse_args()

    # load global parameters from argument
    PROJECT_NAME = args.project_name
    BO_TESTING_MODE = args.test_bo

    config = omegaconf.OmegaConf.load(args.config)
    override_config = omegaconf.OmegaConf.from_dotlist(unknown_args)
    validate_overrides(config, override_config)
    print_introduction(args, config, override_config)
    config = omegaconf.OmegaConf.merge(config, override_config)
    main(config)