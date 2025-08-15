import yaml 
import numpy as np
import pandas as pd
from pprint import pprint
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
		self.feature_to_neuron = {}
		self.neuron_to_feature = {}

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
			cites_file = "data/microseer/microseer.cites"

			with open("data/microseer/microseer.indices", 'r') as file:
				lines = file.readlines()

			self.training_paper_ids.update(lines[1].strip().split(', '))
			self.testing_paper_ids.update(lines[2].strip().split(', '))


		elif config["dataset"] == "miniseer":
			content_file = "data/miniseer/miniseer.content"
			cites_file = "data/miniseer/miniseer.cites"

			with open("data/miniseer/miniseer.indices", 'r') as file:
				lines = file.readlines()

			self.training_paper_ids.update(lines[1].strip().split(', '))
			self.testing_paper_ids.update(lines[2].strip().split(', '))


		elif config["dataset"] == "citeseer":
			content_file = "data/citeseer/citeseer.content"
			cites_file = "data/citeseer/citeseer.cites"

			with open("data/citeseer/citeseer_train_nodes.txt", 'r') as file:
				lines = file.readlines()

			self.training_paper_ids.update(lines[1].strip().split(', '))
			self.testing_paper_ids.update(lines[3].strip().split(', '))


		elif config["dataset"] == "cora":
			content_file = "data/Cora/cora/cora.content"
			cites_file = "data/Cora/cora/cora.cites"

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

			# Create feature neurons and update associated dictionaries
			for i in range(self.num_features):
				feature_neuron_id = self.snn.create_neuron(
					threshold=config["feature_threshold"],
					leak = config["feature_leak"], 
					refractory_period = config["feature_ref"]
				).idx

				# Since feature neurons were created first, their indices are the 
		    	# same as the feature neuron IDs
				# These data structures may not be needed and may be
				# removed in the future
				self.feature_to_neuron[i] = feature_neuron_id
				self.neuron_to_feature[feature_neuron_id] = i


		# Read content_file line by line and create the SNN
		with open(content_file, 'r') as file:
			for line in file:

				# Gather contents of the line
			    line_contents = line.strip().split()


			    # Extract paper and topic IDs
			    paper_id = line_contents[0]
			    topic_id = line_contents[-1]


			    # Extract features
			    features = None

			    if config["features"] == 1:
			    	features = np.nonzero(np.array(line_contents[1:-1], dtype=int))[0]


			    # Update dictionary that maps papers to topics
			    if paper_id not in self.paper_to_topic:
			    	self.paper_to_topic[paper_id] = topic_id

			    else:
			    	raise RuntimeError(f"Topic for paper_id {paper_id} already documented")
			    

			    # Create topic neuron only if it doesn't already exist
			    if topic_id not in self.topic_to_neuron:
			    	topic_neuron_id = self.snn.create_neuron(
			    		threshold = config["topic_threshold"],
			    		leak = config["topic_leak"],
			    	).idx

			    	self.num_topics += 1
			    	self.topic_to_neuron[topic_id] = topic_neuron_id
			    	self.neuron_to_topic[topic_neuron_id] = topic_id
			    	self.topic_to_paper[topic_id] = [paper_id]
			    
			    else:
			    	self.topic_to_paper[topic_id].append(paper_id)


			    # Create training paper neuron and associated synapses
			    paper_neuron_id = None

			    if paper_id in self.training_paper_ids:

			    	# Create training paper neuron
			    	paper_neuron_id = self.snn.create_neuron(
			    		threshold = config["paper_threshold"],
			    		leak = config["paper_leak"],
			    		refractory_period = config["train_ref"]
			    	).idx

			    	# Increment number of papers
			    	self.num_papers += 1

			    	# Create synapse from training paper neuron to topic neuron
			    	self.snn.create_synapse(
			    		pre_id = paper_neuron_id,
			    		post_id = topic_neuron_id,
			    		weight = config["train_to_topic_weight"],
			    		delay = config["train_to_topic_delay"],
			    		stdp_enabled = False
			    	)

			    	# Create synapse from topic neuron to training paper neuron
			    	self.snn.create_synapse(
			    		pre_id = topic_neuron_id,
			    		post_id = paper_neuron_id,
			    		weight = config["topic_to_train_weight"],
			    		delay = config["topic_to_train_delay"],
			    		stdp_enabled = False
			    	)

			    
			    # Create validation/testing paper neuron
			    elif (paper_id in self.validation_paper_ids) or (paper_id in self.testing_paper_ids):
			    	paper_neuron_id = self.snn.create_neuron(
			    		threshold = config["paper_threshold"],
			    		leak = config["paper_leak"],
			    		refractory_period = config["test_ref"]
			    	).idx

			    	self.num_papers += 1

			    	# # Create synapse from validation/testing paper neuron to topic neuron
			    	# self.snn.create_synapse(
			    	# 	pre_id = paper_neuron_id,
			    	# 	post_id = topic_neuron_id,
			    	# 	weight = config["test_to_topic_weight"],
			    	# 	delay = config["test_to_topic_delay"],
			    	# 	stdp_enabled = False
			    	# )

			    	# # Create synapse from topic neuron to validation/testing paper neuron
			    	# self.snn.create_synapse(
			    	# 	pre_id = topic_neuron_id,
			    	# 	post_id = paper_neuron_id,
			    	# 	weight = config["topic_to_test_weight"],
			    	# 	delay = config["topic_to_test_delay"],
			    	# 	stdp_enabled = False
			    	# )


			    else:
			    	raise RuntimeError(f"paper_id {paper_id} is not in training, validation, or test sets")


			    # Update the dictionaries that map papers to paper neurons and vice versa
			    self.paper_to_neuron[paper_id] = paper_neuron_id
			    self.neuron_to_paper[paper_neuron_id] = paper_id


			    # Create synapse from paper neuron to its feature neurons
			    if config["features"] == 1:
				    for feature in features:

			    		# Create synapse from paper neuron to feature neuron 
			    		# Since feature neurons were created first, their indices are the 
			    		# same as the feature neuron IDs
			    		self.snn.create_synapse(
			    			pre_id = paper_neuron_id,
			    			post_id = feature,
			    			weight = config["paper_to_feature_weight"],
			    			delay = config["paper_to_feature_delay"],
			    			stdp_enabled = False
			    		)

			    		# Update paper_to_feature dictionary
			    		if paper_id not in self.paper_to_feature:
				    		self.paper_to_feature[paper_id] = [feature]

				    	else:
				    		self.paper_to_feature[paper_id].append(feature)


				    	# Create synapse from feature neuron to paper neuron
				    	# Since feature neurons were created first, their indices are the 
			    		# same as the feature neuron IDs
				    	self.snn.create_synapse(
				    		pre_id = feature,
				    		post_id = paper_neuron_id,
				    		weight = config["feature_to_paper_weight"],
				    		delay = config["feature_to_paper_delay"]
				    	)


			    		# Update feature_to_paper dictionary
			    		if feature not in self.feature_to_paper:
				    		self.feature_to_paper[feature] = [paper_id]

				    	else:
				    		self.feature_to_paper[feature].append(paper_id)


		# Create synapses from paper neurons to paper neurons
		with open(cites_file, 'r') as file:
			for line in file:
				# Extract the line information
				post_paper_id, pre_paper_id = line.strip().split()

				# Citations in the .cites file are paper2 -> paper1 on each line
				# Refer to the README in the Citeseer dataset 
				# Therefore, synapses must be created from the second paper to the first paper
				# Here, we will create bi-directional synapses

				# Create synapse from pre_paper_id to post_paper_id
				self.snn.create_synapse(
					pre_id = self.paper_to_neuron[pre_paper_id],
					post_id = self.paper_to_neuron[post_paper_id],
					weight = config["graph_weight"],
					delay = config["graph_delay"],
					stdp_enabled = False
				)

				# Create synapse from post_paper_id to pre_paper_id
				self.snn.create_synapse(
					pre_id = self.paper_to_neuron[post_paper_id],
					post_id = self.paper_to_neuron[pre_paper_id],
					weight = config["graph_weight"],
					delay = config["graph_delay"],
					stdp_enabled = False,
					exist = "dontadd"
				)


		# Create STDP synapses from validation and testing paper neurons to topic neurons and vice versa
		for paper_id in self.validation_paper_ids.union(self.testing_paper_ids):
			for topic_id in self.topic_to_neuron:

				# Create synapse from paper neuron to topic neuron
				self.snn.create_synapse(
					pre_id = self.paper_to_neuron[paper_id],
					post_id = self.topic_to_neuron[topic_id],
					weight = config["test_to_topic_weight"],
					delay = config["test_to_topic_delay"],
					stdp_enabled = True
				)

				# Create synapse from topic neuron to paper neuron
				self.snn.create_synapse(
					pre_id = self.topic_to_neuron[topic_id],
					post_id = self.paper_to_neuron[paper_id],
					weight = config["topic_to_test_weight"],
					delay = config["topic_to_test_delay"],
					stdp_enabled = True
				)


		# Create STDP synapses from features to topics and vice versa
		if config["features"] == 1:
			for f in range(self.num_features):
				for t in self.topic_to_neuron:

					# Create synapse from feature to topic
					self.snn.create_synapse(
						pre_id = f,
						post_id = self.topic_to_neuron[t],
						weight = config["feature_to_topic_weight"],
						delay = config["feature_to_topic_delay"],
						stdp_enabled = True
					)

					# Create synapse from topic to feature
					self.snn.create_synapse(
						pre_id = self.topic_to_neuron[t],
						post_id = f,
						weight = config["topic_to_feature_weight"],
						delay = config["topic_to_feature_delay"],
						stdp_enabled = True
					)

		# Create (inhibitory) STDP synapses from features to features??
		# for e in range(self.num_features):
		# 	for f in range(e+1, self.num_features):
		# 		# Create inhibitory synapse from e to f
		# 		self.snn.create_synapse(
		# 			pre_id = e,
		# 			post_id = f,
		# 			weight = config["feature_to_feature_weight"],
		# 			delay = config["feature_to_feature_delay"],
		# 			stdp_enabled = True
		# 		)

		# 		# Create inhibitory synapse from f to e
		# 		self.snn.create_synapse(
		# 			pre_id = f,
		# 			post_id = e,
		# 			weight = config["feature_to_feature_weight"],
		# 			delay = config["feature_to_feature_delay"],
		# 			stdp_enabled = True
		# 		)


		# STDP setup
		self.snn.stdp_setup(
			Apos = config["apos"],
			Aneg = config["aneg"] * len(config["apos"]),
			positive_update = True,
			negative_update = True
		)




	def predict(self, verbose="True"):
		""" Predicts the topic for a given paper

		"""

		# Check if verbose is True or False
		if not ((verbose != True) and (verbose != False)):
			raise ValueError(f"verbose must be {True} or {False}")


		# Validation or test mode
		if config["mode"] == "validation":
			prediction_paper_ids = self.validation_paper_ids

		elif config["mode"] == "test":
			prediction_paper_ids = self.testing_paper_ids

		else:
			raise ValueError(f"Unknown config mode {config["mode"]}")


		# Reset SNN
		self.snn.reset()

		print("\n")


		# Start testing
		topic_neuron_ids = list(self.neuron_to_topic)

		print(f"Topic neuron IDs: {topic_neuron_ids}")
		print()

		outputs = {
			"PaperID": [],
			"GroundTruth": [],
			"Prediction": [],
			"IsCorrect": []
		}

		accuracy = 0

		for paper_id in prediction_paper_ids:

			# Add spike to the paper neuron
			print(f"Adding spike to {self.paper_to_neuron[paper_id]}")

			self.snn.add_spike(
				neuron_id = self.paper_to_neuron[paper_id],
				time = 0,
				value = 1000
			)

			print(f"Neuron threshold: {self.snn.neuron_thresholds[self.paper_to_neuron[paper_id]]}")

			# Simulate
			self.snn.simulate(config["simtime"])

			# Predict by virtue of maximum spiking topic neuron
			prediction_index = np.array(self.snn.spike_train)[:, topic_neuron_ids].sum(axis=0).argmax()
			prediction = self.neuron_to_topic[topic_neuron_ids[prediction_index]]
			accuracy += int(self.paper_to_topic[paper_id] == prediction) 

			# Gather Outputs
			outputs["PaperID"].append(paper_id)
			outputs["GroundTruth"].append(self.paper_to_topic[paper_id])
			outputs["Prediction"].append(prediction)
			outputs["IsCorrect"].append(self.paper_to_topic[paper_id] == prediction)

			# Print spike train
			self.snn.print_spike_train()
			print()

			# Reset
			self.snn.reset()
		
		# Print outputs
		if verbose:
			print(pd.DataFrame(outputs))

		# Print accuracy 
		accuracy = accuracy / len(prediction_paper_ids)
		print(f"\nAccuracy: {accuracy}")

		return accuracy



if __name__ == "__main__":

	# Read config file
	config = yaml.safe_load(open(DATASET_CONFIG))

	# Create SGNN object
	sgnn = SGNN(config)

	# Load dataset
	sgnn.load_dataset()


	print(f"\nNum training papers: {sgnn.num_training_papers}")
	print(f"\nNum validation papers: {sgnn.num_validation_papers}")
	print(f"\nNum testing papers: {sgnn.num_testing_papers}")
	
	print()

	print(f"\nTraining paper IDs:")
	pprint(sgnn.training_paper_ids)

	print(f"\nValidation paper IDs:")
	pprint(sgnn.validation_paper_ids)

	print(f"\nTesting paper IDs:")
	pprint(sgnn.testing_paper_ids)
	
	print()

	print(f"\nNum papers: {sgnn.num_papers}")
	print(f"\nNum topics: {sgnn.num_topics}")
	print(f"\nNum features: {sgnn.num_features}")
	
	print()

	print("\npaper_to_neuron:")
	pprint(sgnn.paper_to_neuron)

	print("\nneuron_to_paper:") 
	pprint(sgnn.neuron_to_paper)

	print("\ntopic_to_neuron:")
	pprint(sgnn.topic_to_neuron)

	print("\nneuron_to_topic:") 
	pprint(sgnn.neuron_to_topic)

	print("\nfeature_to_neuron:") 
	pprint(sgnn.feature_to_neuron)

	print("\nneuron_to_feature:") 
	pprint(sgnn.neuron_to_feature)

	print("\npaper_to_topic:") 
	pprint(sgnn.paper_to_topic)

	print("\ntopic_to_paper:") 
	pprint(sgnn.topic_to_paper)

	print("\npaper_to_feature:") 
	pprint(sgnn.paper_to_feature)

	print("\nfeature_to_paper:") 
	pprint(sgnn.feature_to_paper)

	print("\n")

	print(sgnn.snn)


	# Test dataset
	sgnn.predict()


