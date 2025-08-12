import yaml 
import pandas
from superneuromat import SNN


DATASET_CONFIG = "configs/microseer/default_microseer_config.yaml"


class SGNN:
	""" Manage citation networks data, create spiking graph neural networks (SGNNs)

	"""

	def __init__(self, config):
		""" Initialize all class variables

		"""

		# Config
		self.config = config

		# Count variables
		self.num_training_papers = 0
		self.num_validation_papers = 0
		self.num_testing_papers = 0

		# Mapping variables: IDs
		self.paper_to_neuron = {}
		self.neuron_to_paper = {}
		self.topic_to_neuron = {}
		self.neuron_to_topic = {}
		self.feature_to_neuron = {}
		self.neuron_to_feature = {}

		# Mapping variables: papers, topics, and features
		self.paper_to_topic = {}
		self.topic_to_paper = {}
		self.paper_to_feature = {}
		self.feature_to_paper = {}

		# Training, validation, and testing papers
		self.training_paper_ids = []
		self.validation_paper_ids = []
		self.testing_paper_ids = []

		# SuperNeuroMAT SNN 
		self.snn = SNN()



	def load_dataset(self):
		""" Selects the appropriate load function for citation network dataset

		"""

		# Choose file path file path based on the dataset
		if config["dataset"] == "microseer":
			content_file = "data/microseer/microseer.indices"
			cites_file = "data/microseer/microseer/cites"

		elif config["dataset"] == "miniseer":
			content_file = "data/miniseer/miniseer.indices"
			cites_file = "data/miniseer/miniseer/cites"

		elif config["dataset"] == "citeseer":
			content_file = "data/citeseer/citeseer.indices"
			cites_file = "data/citeseer/citeseer/cites"

		elif config["dataset"] == "cora":
			content_file = "data/cora/cora.indices"
			cites_file = "data/cora/cora/cites"

		elif config["dataset"] == "pubmed":
			content_file = "data/cora/cora.indices"
			cites_file = "data/cora/cora/cites"

		# To add: pubmed and biteseer

		else:
			raise ValueError("Unknown dataset in config file")


		# Read content_file



	# def load_microseer(self):
	# 	""" Loads the microseer dataset

	# 	"""

	# 	# *** LOAD PAPERS ***
	# 	with open("data/microseer/microseer.indices") as f:
	# 		lines = f.readlines()


	# 	# Extract indices of training and testing papers
	# 	self.training_paper_ids = lines[1].strip().split()
	# 	self.testing_paper_ids = lines[2].strip().split()


	# 	# Create training neurons
	# 	for paper_id in training_paper_ids:
	# 		neuron_id = self.snn.create_neuron().idx

	# 		if paper_id not in self.paper_to_neuron:
	# 			self.paper_to_neuron[paper_id] = neuron_id

	# 		else:
	# 			raise RuntimeError(f"Paper ID {paper_id} already exists in paper_to_neuron")

	# 		if neuron_id not in self.neuron_to_paper:
	# 			self.neuron_to_paper[neuron_id] = paper_id

	# 		else:
	# 			raise RuntimeError("Neuron ID already exists in paper_to_neuron")


	# 	# Create testing neurons
	# 	for paper_id in testing_paper_ids: 
	# 		neuron_id = self.snn.create_neuron().idx
			
	# 		if paper_id not in self.paper_to_neuron:
	# 			self.paper_to_neuron[paper_id] = neuron_id

	# 		else:
	# 			raise RuntimeError("Paper ID already exists in paper_to_neuron")

	# 		if neuron_id not in self.neuron_to_paper:
	# 			self.neuron_to_paper[neuron_id] = paper_id

	# 		else:
	# 			raise RuntimeError("Neuron ID already exists in paper_to_neuron")


	# 	# *** LOAD TOPICS AND FEATURES ***
	# 	with open("data/microseer/microseer.content") as f:
	# 		lines = f.readlines()


	# 	# 

	# 	print(self.snn)



if __name__ == "__main__":

	# Read config file
	config = yaml.safe_load(open(DATASET_CONFIG))

	# Create SGNN object
	sgnn = SGNN(config)

	# Load dataset
	sgnn.load_dataset()

