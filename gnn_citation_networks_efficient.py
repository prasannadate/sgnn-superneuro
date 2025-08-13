import yaml 
import pandas
from superneuromat import SNN


DATASET_CONFIG = "configs/microseer/default_microseer_config.yaml"
# DATASET_CONFIG = "configs/miniseer/default_miniseer_config.yaml"
# DATASET_CONFIG = "configs/citeseer/default_citeseer_config.yaml"
# DATASET_CONFIG = "configs/cora/default_cora_config.yaml"


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
		self.num_papers = 0 
		self.num_topics = 0
		self.num_features = 0

		# Training, validation, and testing paper IDs
		self.training_paper_ids = set()
		self.validation_paper_ids = set()
		self.testing_paper_ids = set()

		# Mapping variables: IDs
		self.paper_to_neuron = {}
		self.neuron_to_paper = {}
		self.topic_to_neuron = {}
		self.neuron_to_topic = {}
		# self.feature_to_neuron = {}
		# self.neuron_to_feature = {}

		# Mapping variables: papers, topics, and features
		self.paper_to_topic = {}
		self.topic_to_paper = {}
		self.paper_to_feature = {}
		self.feature_to_paper = {}

		# SuperNeuroMAT SNN 
		self.snn = SNN()



	def load_dataset(self):
		""" Selects the appropriate load function for citation network dataset

		"""

		# Choose file path file path based on the dataset
		if config["dataset"] == "microseer":
			content_file = "data/microseer/microseer.content"
			cites_file = "data/microseer/microseer/cites"

			with open("data/microseer/microseer.indices", 'r') as file:
				lines = file.readlines()

			self.training_paper_ids.update(lines[1].strip().split(', '))
			self.testing_paper_ids.update(lines[2].strip().split(', '))


		elif config["dataset"] == "miniseer":
			content_file = "data/miniseer/miniseer.content"
			cites_file = "data/miniseer/miniseer/cites"

			with open("data/miniseer/miniseer.indices", 'r') as file:
				lines = file.readlines()

			self.training_paper_ids.update(lines[1].strip().split(', '))
			self.testing_paper_ids.update(lines[2].strip().split(', '))


		elif config["dataset"] == "citeseer":
			content_file = "data/citeseer/citeseer.content"
			cites_file = "data/citeseer/citeseer/cites"

			with open("data/citeseer/citeseer_train_nodes.txt", 'r') as file:
				lines = file.readlines()

			self.training_paper_ids.update(lines[1].strip().split(', '))
			self.testing_paper_ids.update(lines[3].strip().split(', '))


		elif config["dataset"] == "cora":
			content_file = "data/Cora/cora/cora.content"
			cites_file = "data/Cora/cora/cora/cites"

			with open("data/Cora/cora_train_indices.txt", 'r') as file:
				lines = file.readlines()

			self.training_paper_ids.update(lines[1].strip().split(', '))
			self.testing_paper_ids.update(lines[3].strip().split(', '))

		
		# To add: pubmed and biteseer

		
		else:
			raise ValueError("Unknown dataset in config file")

		
		# Count number of papers in training and testing sets
		self.num_training_papers = len(self.training_paper_ids)
		self.num_testing_papers = len(self.testing_paper_ids)


		# Count number of features and create feature neurons
		if config["features"] == 1:
			with open(content_file, 'r') as file:
				line = file.readline().strip().split()

			self.num_features = len(line) - 2

			for i in range(self.num_features):
				neuron_id = self.snn.create_neuron(
					threshold=config["feature_threshold"],
					leak = config["feature_leak"], 
					refractory_period = config["feature_ref"]
				)

				# self.feature_to_neuron[i] = neuron_id
				# self.neuron_to_feature[neuron_id] = i


		# Read content_file line by line and create the SNN
		with open(content_file, 'r') as file:
			for line in file:

				# Gather contents of the line
			    line_contents = line.strip().split()


			    # Extract paper and topic IDs
			    paper_id = line_contents[0]
			    topic_id = line_contents[-1]


			    # Update dictionary that maps papers to topics and vice versa
			    if paper_id not in self.paper_to_topic:
			    	self.paper_to_topic[paper_id] = topic_id

			    else:
			    	raise RuntimeError(f"Topic for paper_id {paper_id} already documented")

			    if topic_id not in self.topic_to_paper:
			    	self.topic_to_paper[topic_id] = [paper_id]

			    else:
			    	self.topic_to_paper[topic_id].append(paper_id)
			    

			    # Create training paper neuron
			    if paper_id in self.training_paper_ids:
			    	neuron_id = self.snn.create_neuron(
			    		threshold = config["paper_threshold"],
			    		leak = config["paper_leak"],
			    		refractory_period = config["train_ref"]
			    	)
			    
			    # Create validation paper neuron
			    elif paper_id in self.validation_paper_ids:
			    	neuron_id = self.snn.create_neuron(
			    		threshold = config["paper_threshold"],
			    		leak = config["paper_leak"],
			    		refractory_period = config["validation_ref"]
			    	)

			    # Create test paper neuron
			    elif paper_id in self.testing_paper_ids:
			    	neuron_id = self.snn.create_neuron(
			    		threshold = config["paper_threshold"],
			    		leak = config["paper_leak"],
			    		refractory_period = config["test_ref"]
			    	)

			    else:
			    	raise RuntimeError(f"paper_id {paper_id} is not in training, validation, or test sets")
			    	# neuron_id = self.snn.create_neuron()


			    # Update the dictionaries that map papers to paper neurons and vice versa
			    self.paper_to_neuron[paper_id] = neuron_id
			    self.neuron_to_paper[neuron_id] = paper_id


			    # Create topic neuron only if it doesn't already exist
			    if topic_id not in self.topic_to_neuron:
			    	neuron_id = self.snn.create_neuron()
			    	self.topic_to_neuron[topic_id] = neuron_id
			    	self.neuron_to_topic[neuron_id] = topic_id




			    # Create synapses from paper neurons to topic neurons


			    # Create synapses from paper neurons to feature neurons


			    # Update dictionary that maps papers to features and vice versa


			    # Create synapses from paper neurons to paper neurons





if __name__ == "__main__":

	# Read config file
	config = yaml.safe_load(open(DATASET_CONFIG))

	# Create SGNN object
	sgnn = SGNN(config)

	# Load dataset
	sgnn.load_dataset()

	print(f"\nNum training papers: {sgnn.num_training_papers}")
	print(f"\nNum testing papers: {sgnn.num_testing_papers}")
	print(f"\nTraining paper IDs: {sgnn.training_paper_ids}")
	print()

	print(f"\nNum papers: {sgnn.num_papers}")
	print(f"\nNum topics: {sgnn.num_topics}")
	print(f"\nNum features: {sgnn.num_features}")
	print()

	print(f"\npaper_to_neuron: {sgnn.paper_to_neuron}")
	print(f"\nneuron_to_paper: {sgnn.neuron_to_paper}")
	print(f"\ntopic_to_neuron: {sgnn.topic_to_neuron}")
	print(f"\nneuron_to_topic: {sgnn.neuron_to_topic}")
	# print(f"\nfeature_to_neuron: {sgnn.feature_to_neuron}")
	# print(f"\nneuron_to_feature: {sgnn.neuron_to_feature}")
	print(f"\npaper_to_topic: {sgnn.paper_to_topic}")
	print(f"\ntopic_to_paper: {sgnn.topic_to_paper}")

	# print(sgnn.snn)

