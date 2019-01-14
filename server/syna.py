import numpy as np
from flask import Response
from neuralnetwork import *

class Backend():

	# Hyper parameters
	dataset = []
	
	layer_dims = [3, 5, 5, 5, 5, 5, 3] # list
	learning_rate = 1.9
	iterations = 10000 # for each dataset
	train = True
	max_dataset_size = 1000
	#activations = ['relu'] * len(layer_dims)
	#activations = ['sigmoid'] * len(layer_dims)
	activations = ['tanh'] * len(layer_dims)

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
				AL, caches = L_model_forward(X, parameters, self.activations)

				#print ('X = ' + str(X) + '\n')
				#print ('AL = ' + str(AL) + '\n')
				#print ('Y = ' + str(Y) + '\n')

				grads = L_model_backward(AL, Y, caches, self.activations)
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
		AL, caches = L_model_forward(X, parameters, self.activations)
		print ('X = ' + str(X) + '\n')
		print ('Y = ' + str(Y) + '\n')
		print ('AL = ' + str(AL) + '\n')

	def process_dataset(self, data):
		# dataset input: [i0, i1 ... miN_X] # m is number of training examples 
		# dataset output [o0, o1 ... moN_Y]

		#curl -X POST 127.0.0.1:5002/dataset --header "Content-Type: application/json" --data '{"dataset":{"inputs":["beep", "boop"], "outputs":["o1", "o2"]}}'
	
		inputs = data['dataset']['inputs']
		outputs= data['dataset']['outputs']
		#print 'in = ' + str(inputs) + ' ... out = ' + str(outputs) + '\n'

		if (len(inputs) % self.layer_dims[0] != 0 or len(outputs) % self.layer_dims[len(self.layer_dims)] != 0):
			return Response('Error:  data inputs or outputs mismatch...\n', 400)

		if (len(inputs) / self.layer_dims[0] != len(outputs) / self.layer_dims[len(self.layer_dims)]):
			return Response('Error:  number of inputs != number of outputs', 400)

		while len(self.dataset) >= self.max_dataset_size:
			del self.dataset[0]

		for i in range(0, (len(inputs) / self.layer_dims[0])):
			X = inputs[(i * self.layer_dims[0]):((i + 1) * self.layer_dims[0])]
			Y = outputs[(i * self.layer_dims[len(self.layer_dims)]):((i + 1) * self.layer_dims[len(self.layer_dims)])]
			self.dataset.append((X, Y))

		return Response('Sucess:  dataset processed', 200)

	def modify_neuralnet(self, data):
		resp = ''
		# layer_dims
		if ('neuralnet' not in data):
			return Response('Error: neural net needed', 400)
		if ('layer_dims' in data['neuralnet']):
			layd = data['neuralnet']['layer_dims']
			if (len(layd) < 2): # needs at least an input and output layer
				return Response('Error:  layer_dims invalid', 400)
			# check each value is valid integer
			for i in layd:
				if ((not isinstance(i, int)) or (i <= 0)):
					return Response('Error: invalid value for layer_dims. ' + str(i), 400)
			self.layer_dims = layd
			resp += 'new layer_dims = ' + str(self.layer_dims) + '\n'

		# iterations
		if ('iterations' in data['neuralnet']):
			iters = data['neuralnet']['iterations']
			if ((not isinstance(iters, int)) or (iters <= 0)):
				return Response('Error: invalid value for iterations. ' + str(iters), 400)
			self.iterations = iters
			resp += 'new iterations = ' + str(self.iterations) + '\n'

		# max_dataset_size
		if('max_dataset_size' in data['neuralnet']):
			dset = data['neuralnet']['max_dataset_size']
			if ((not isinstance(iters, int)) or (iters <= 0)):
				return Response('Error: invalid value for max data set size. ' + str(dset), 400)
			self.max_dataset_size = dset
			#self.dataset = [] # trim current dataset?
			resp += 'new max_dataset_size = ' + str(self.max_dataset_size) + '\n'

		# activations
		if ('activations' in data['neuralnet']):
			acts = data['neuralnet']['activations']
			if ((len(acts) != 1) or (len(acts) != len(self.layer_dims))):
				return Response('Error: activations need to match up to layer_dims or just be 1 value', 400)
			#for (a in acts):
			#	if (a != 'tanh' or a != 'linear' or a != 'sigmoid' or a != 'relu')
			#		return Response('Error:  invalid activation provided ' + str(a), 400)

			if len(acts == 1):
				self.activations = [acts] * len(self.layer_dims)
			else:
				self.activations = acts
			resp += 'new activations = ' + str(self.activations) + '\n'

		# learning_rate
		if('learning_rate' in data['neuralnet']):
			lrate = data['neuralnet']['learning_rate']
			if (lrate <= 0):
				return Response('Error: invalid value for iterations. ' + str(lrate), 400)
			self.learning_rate = lrate
			resp += 'new learning_rate = ' + str(lrate) + '\n'

		return Response('Success: neuralnet updated. ' + resp, 202)



	def generate_test_data(self):
		# generates training data using sin(x)s.  
		X = np.random.randn(3, 1000) * 0.01
		Y = np.sin(X * np.pi)
		return (X, Y)
