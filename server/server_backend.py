import numpy as np
from neuralnetwork import *

class Backend():

	# Hyper parameters
	dataset = []
	n_x = 2
	n_h = 5 # tuple of hidden layers with number of perceptrons
	n_y = 2
	learning_rate = 0.9
	iterations = 10 # for each dataset
	train = True
	max_dataset_size = 10000

	# separate process
	def start_training(self):
		
		# init nn
		layer_dims = [self.n_x, self.n_h, self.n_y] # list
		parameters = initialize_parameters_deep(layer_dims)
		
		while self.train:
			# copy dataset into numpy values
			#X = np.array([x for x,_ in self.dataset])
			#print 'X: ' + str(X) + '\n'
			#Y = np.array([y for _,y in self.dataset])
			#print 'Y: ' + str(Y) + '\n'
			X = np.array([[2, 3], [4,9]]).T
			Y = np.array([[0.3, 0.7],[0.4, 0.1]]).T
			print ('X.shape = ' + str(X.shape) + ' ... Y.shape = ' + str(Y.shape) + '\n')
			self.train = False

			for i in range(0, self.iterations):
				AL, caches = L_model_forward(X, parameters, "sigmoid")

				cost = compute_cost(AL, Y)

				grads = L_model_backward(AL, Y, caches, "sigmoid")
				#print ('grads = ' + str(grads) + '\n')

				parameters = update_parameters(parameters, grads, self.learning_rate)
				#print ('params = ' + str(parameters) + '\n')

				if i % 1000: #(i + 1) == self.iterations:
					print 'cost = ' + str(cost) + '\n'


	def process_data(self, data):
		# dataset input: [i0, i1 ... miN_X] # m is number of training examples 
		# dataset output [o0, o1 ... moN_Y]

		#curl -X POST 127.0.0.1:5002/dataset --header "Content-Type: application/json" --data '{"dataset":{"inputs":["beep", "boop"], "outputs":["o1", "o2"]}}'
	
		inputs = data['dataset']['inputs']
		outputs= data['dataset']['outputs']
		#print 'in = ' + str(inputs) + ' ... out = ' + str(outputs) + '\n'

		if (len(inputs) % self.n_x != 0 or len(outputs) % self.n_y != 0):
			return False # TODO: more info to send REST response

		if (len(inputs) / self.n_x != len(outputs) / self.n_y):
			return False # TODO: more info response

		while len(self.dataset) >= self.max_dataset_size:
			del self.dataset[0]

		for i in range(0, (len(inputs) / self.n_x)):
			X = inputs[(i * self.n_x):((i + 1) * self.n_x)]
			Y = outputs[(i * self.n_y):((i + 1) * self.n_y)]
			self.dataset.append((X, Y))

		return True
