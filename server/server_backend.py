import numpy as np
from neuralnetwork import *

class Backend():

	# Hyper parameters
	dataset = []
	n_x = 17
	n_h = 50
	n_y = 16
	learning_rate = 0.9
	iterations = 100 # for each dataset
	train = True

	# separate process
	def start_training(self):
		
		# init nn
		layer_dims = (n_x, n_h, n_y)
		parameters = initialize_parameters_deep(layer_dims)
		
		while train:
			# copy dataset into numpy values
			X = np.array([x for x,_ in dataset])
			Y = np.array([y for _,y in dataset])

			for i in range(0, iterations):
				AL, caches = L_model_forward(X, parameters)

				cost = compute_cost(AL, Y)

				grads = L_model_backward(AL, Y, caches)

				parameters = update_parameters(parameters, grads, learning_rate)

				if i == iterations:
					print 'cost = ' + cost + '\n'


	def validate_data(self, data):
		# dataset input: [i0, i1 ... miN_X] # m is number of training examples 
		# dataset output [o0, o1 ... moN_Y]

		#curl -X POST 127.0.0.1:5002/dataset --header "Content-Type: application/json" --data '{"dataset":{"inputs":["beep", "boop"], "outputs":["o1", "o2"]}}'
	
		for x in data['dataset']:
			print 'valid: ' + x + '\n'
		return True
