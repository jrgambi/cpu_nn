#TODO: dataset manages dataset used in neural network
import numpy as np

MAX = 10000
dataset = []# tuple of inputs, outputs


def load_dataset():
	# memory, maintains one and makes a copy for neuralnet to train with
	# maintain other copy in file system?
	#print 'load dataset\n'
	return dataset

def append_data(data):
	# check data is correct size
	while (dataset and dataset.len > (MAX - data.len)):
	   del dataset[0]
	dataset.append(data)
	print 'append: ' + str(data) + '\n' 

	# file system stuff
