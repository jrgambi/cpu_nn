#TODO: dataset manages dataset used in neural network
import numpy as np

dataset = [] # tuple of inputs, outputs
MAX = 10000

def load_dataset():
	# memory, maintains one and makes a copy for neuralnet to train with
	# maintain other copy in file system?
	#print 'load dataset\n'
	return dataset

def append_data(data):
	# check data is correct size
	while (dataset.shape[1] >= (MAX - data.shape[1]):
	   dataset = np.remove(dataset, 0, 0))
	np.append(dataset, data)
	#print 'append data\n'

# file system stuff
