# standard pip imports
from multiprocessing import Pool
import numpy as np
import omegaconf

# project specific pip imports
from lmao.solver import BOSolver
from superneuromat.neuromorphicmodel import NeuromorphicModel

# local imports
from graph_data import GraphData
from sgnn import (
    test_paper,
)
from utils import (
    parse_args,
    plot_confusion_matrix,
    print_introduction,
    print_paper_spikes,
    validate_overrides,
)


def solve_gnn(args) -> float:
    graph = GraphData(args.dataset, config)    
    
    if (config["monitors"] == True):
        for i in range(0, len(graph.validation_papers), 20):
            paper = graph.validation_papers[i]
            x = [paper, graph, config]
            test_paper(x)
        sys.exit()

     # Spike each of the test papers in simulation
    i = 0
    correct = 0
    total = 0

    pool = Pool(args.processes)
    if args.mode == "validation":
        papers = []
        for paper in graph.validation_papers:
            papers.append([paper, graph, config])
    
        # predictions will be a list of tuples: (paper, predicted_topic, retval)
        predictions = pool.map(test_paper, papers)
    
        # You already print accuracy using retval
        accuracy = np.mean([ret for (_, _, ret) in predictions])
        print("Validation Accuracy:", accuracy)

        pool.close()

        # Evaluate with confusion matrix, classification report, etc.
        evaluate_predictions(graph, predictions)
 
    if args.mode == "test":
        papers = []
        for paper in graph.test_papers:
            papers.append([paper, graph, config])
        x = pool.map(test_paper, papers) 
        print("Testing Accuracy:", np.sum(x) / len(graph.test_papers))

        pool.close()

def main(config: omegaconf.DictConfig) -> None:
    """
    Main function to execute the script.

    Args:
        config (omegaconf.DictConfig): Merged configuration.
    """

    solve_gnn(config)

    # np.random.seed(args.seed)

    # NUM_PROCESSES = args.processes

    # search_space = Space([
    #     Categorical([10, 20, 30, 40, 50], name="paper_tau_minus"),
    # ])

    # num_initial_points: int = 3

    # optimizer_config = DictConfig({
    #     "num_initial_points": num_initial_points,
    #     "max_iterations": 10,
    #     "num_processes": 1,
    #     "num_repeats": 1,
    #     "optimizer_class": "gp-cpu",
    #     "optimizer": {
    #         "num_initial_points": num_initial_points,
    #     },
    #     "seed": 10,
    # })

    # solver = BOSolver(optimizer_config)
    # result = solver.solve(
    #     solve_test,
    #     use_lp=False,
    #     search_space=search_space
    # )



if __name__ == "__main__":
    args, unknown_args = parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    override_config = omegaconf.OmegaConf.from_dotlist(unknown_args)
    validate_overrides(config, override_config)
    print_introduction(args, config, override_config)
    config = omegaconf.OmegaConf.merge(config, override_config)
    main(config)
