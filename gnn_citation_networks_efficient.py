import yaml 
import pandas
from superneuromat import SNN


DATASET_CONFIG = "configs/microseer/default_microseer_config.yaml"


def load_paper_neurons(config, snn):
	""" Loads the paper info and creates paper neurons in the snn

		Args:
			config: Configuration of the dataset from the yaml file
			snn: SuperNeuroMAT SNN

		Returns:
			paper_to_neuron: Dictionary mapping paper IDs to neuron IDs
			neuron_to_paper: Dictionary mapping neuron IDs to paper IDs

	"""

	if config["dataset"] == "microseer":
		with open("data/microseer/microseer.content")





if __name__ == "__main__":

	# Read config file
	config = yaml.safe_load(open(DATASET_CONFIG))

	# Create SNN
	snn = SNN()

	# Read paper data
	load_paper_neurons(config, snn)

	# Read topics and features
	# load_topics_and_features(config, snn)

