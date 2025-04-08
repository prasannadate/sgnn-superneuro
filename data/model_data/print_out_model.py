import superneuromat
import pickle


with open("pre_sim_model.pkl", "rb") as f:
    model = pickle.load(f)

#example ways to access the model data
#I would check the github source for the variable names if I did not include anything that you want
print(model)
print(model.synaptic_weights)
print(model.synaptic_delays)
print(model.neuron_thresholds)
