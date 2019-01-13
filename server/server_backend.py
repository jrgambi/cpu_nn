import numpy as np
from neuralnetwork import *

class Backend():

	# Hyper parameters
	dataset = []
	
	layer_dims = [3, 5, 5, 5, 5, 5, 3] # list
	learning_rate = 1.9
	iterations = 10000 # for each dataset
	train = True
	max_dataset_size = 1000
	#activation = "relu" 
	#activation = "sigmoid"
	activation = "tanh"

	# separate process
	def start_training(self):

		np.random.seed()
	
		# init nn
		parameters = initialize_parameters_deep(self.layer_dims)
		print ('starting params = ' + str(parameters) + '\n')
		while self.train:
			# copy dataset into numpy values
			#X = np.array([x for x,_ in self.dataset])
			#print 'X: ' + str(X) + '\n'
			#Y = np.array([y for _,y in self.dataset])
			#print 'Y: ' + str(Y) + '\n'
			(X, Y) = self.generate_data()
			#print ('X.shape = ' + str(X.shape) + ' ... Y.shape = ' + str(Y.shape) + '\n')
			self.train = False

			for i in range(0, self.iterations):
				AL, caches = L_model_forward(X, parameters, self.activation)

				#print ('X = ' + str(X) + '\n')
				#print ('AL = ' + str(AL) + '\n')
				#print ('Y = ' + str(Y) + '\n')

				grads = L_model_backward(AL, Y, caches, self.activation)
				#print ('grads = ' + str(grads) + '\n')

				parameters = update_parameters(parameters, grads, self.learning_rate)
				#print ('params = ' + str(parameters) + '\n')

				if i % 1000 == 0: #(i + 1) == self.iterations:
					cost = compute_cost(AL, Y)
					print 'cost = ' + str(cost) + '\n'
					#p = predict(parameters, X, self.activation)
					#print ('prediction = ' + str(p) + '\n')

		X = np.random.randn(3,1) *0.01
		Y = np.sin(X * np.pi)
		AL, caches = L_model_forward(X, parameters, self.activation)
		print ('X = ' + str(X) + '\n')
		print ('Y = ' + str(Y) + '\n')
		print ('AL = ' + str(AL) + '\n')

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

	def generate_data(self):
		# generates training data using sin(x)s.  
		X = np.random.randn(3, 1000) * 0.01
		Y = np.sin(X * np.pi)
		return (X, Y)
