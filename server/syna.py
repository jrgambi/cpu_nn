#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from flask import Response
import json
import threading
from neuralnetwork import *

class Backend():

	# Hyper parameters
	dataset = ([],[]) # tuple of two numpy arrays of same length
	
	layer_dims = [3, 5, 5, 5, 3] # list
	learning_rate = 0.7
	iterations = 10000 # for each dataset
	train = False
	dinit = False # = have parameters been initialized?
	max_dataset_size = 10000
	#activations = ['relu'] * len(layer_dims)
	#activations = ['sigmoid'] * len(layer_dims)
	activations = ['tanh'] * len(layer_dims)
	parameters = []

	# separate process
	def start_training(self, finit=False): # finit = force init

		if finit or not self.dinit: # 
			np.random.seed()
			self.parameters = initialize_parameters_deep(self.layer_dims)
			self.dinit = True

		#print ('starting params = ' + str(parameters) + '\n')
		while self.train:
			# copy dataset into numpy values
			self.dataset = self.generate_test_data() 
			(X, Y) = self.dataset

			#self.train = False

			for i in range(0, self.iterations):
				AL, caches = L_model_forward(X, self.parameters, self.activations)

				#print ('X = ' + str(X) + '\n')
				#print ('AL = ' + str(AL) + '\n')
				#print ('Y = ' + str(Y) + '\n')

				grads = L_model_backward(AL, Y, caches, self.activations)
				#print ('grads = ' + str(grads) + '\n')

				self.parameters = update_parameters(self.parameters, grads, self.learning_rate)
				#print ('params = ' + str(parameters) + '\n')

				if i % 1000 == 0: #(i + 1) == self.iterations:
					cost = compute_cost(AL, Y)
					print ('cost = ' + str(cost) + '\n')
					
					#p = predict(parameters, X, self.activation)
					#print ('prediction = ' + str(p) + '\n')
				if not self.train:
					break

			# test value for generate_test_data
			X = np.random.rand(3,1) #* 0.01
			Y = np.sin(X * np.pi)
			AL, caches = L_model_forward(X, self.parameters, self.activations)
			print ('X = ' + str(X) + '\n')
			print ('Y = ' + str(Y) + '\n')
			print ('AL = ' + str(AL) + '\n')

	def start_training_thread(self, finit=False):
		if not self.train:
			self.train = True
			t1 = threading.Thread(target=self.start_training, args=(finit,))
			t1.start()
		else:
			return Response('already start\n', 200)
			
		return Response('Process started\n', 200)

	# datasets has a many to one relation with neural network?  or 1-to-1?  
	def get_dataset(self, guid=None): # TODO: use headers to specifiy/track current neural network
		# nn_guid will eventually be required
		(X, Y) = self.dataset
		resp = {'dataset':{'inputs':X, 'outputs':Y}}
		if guid is None:
			return Response(json.dumps(resp), 200, mimetype='application/json')
		else: # TODO: DB select using guid?
			return Response(json.dumps(resp), 200, mimetype='application/json')


	def handle_dataset(self, data):
		# dataset input: [i0, i1 ... miN_X] # m is number of training examples 
		# dataset output [o0, o1 ... moN_Y]

		#curl -X POST 127.0.0.1:5002/dataset --header "Content-Type: application/json" --data 
		#	'{"dataset":{"inputs":[[1,2,3], [4,5,6]], "outputs":[[6,5,4], [3,2,1]]}}'

		# validate request
		if 'dataset' not in data or 'inputs' not in data['dataset'] or 'outputs' not in data['dataset']:
			return Response('Error: Request invalid', 400)
	
		inputs = data['dataset']['inputs']
		outputs = data['dataset']['outputs']

		if 'force_new' in data['dataset'] and data['dataset']['force_new']:
			self.dataset = ([],[])

		(X, Y) = self.dataset

		# validate inputs and outputs
		if len(inputs) != len(outputs):
			return Response('Error: invalid data', 400)

		# append values to X and Y and update dataset
		for i in range(0, len(inputs)):
			if ((len(inputs[i]) != self.layer_dims[0]) or (len(outputs[i]) != self.layer_dims[len(self.layer_dims) - 1])):
				return Response('Error: invalid data', 400)
			X.append(inputs[i])
			Y.append(outputs[i])

		self.dataset = (X, Y)

		return Response('Sucess:  dataset processed', 200)

	def modify_neuralnet(self, data):
		resp = ''
		# layer_dims
		if ('neuralnet' not in data):
			return Response('Error: neural net needed', 401)

		if ('hyper_params' not in data['neuralnet']):
			return Response('Error: hyper params needed', 402)

		if ('layer_dims' in data['neuralnet']['hyper_params']):
			layd = data['neuralnet']['hyper_params']['layer_dims']
			if (len(layd) < 2): # needs at least an input and output layer
				return Response('Error:  layer_dims invalid', 403)
			# check each value is valid integer
			for i in layd:
				if ((not isinstance(i, int)) or (i <= 0)):
					return Response('Error: invalid value for layer_dims. ' + str(i), 404)
			if ((layd[0] != self.layer_dims[0]) or (layd[len(layd) - 1] != self.layer_dims[len(self.layer_dims) - 1])):
				self.dataset = [] # reset dataset
			self.layer_dims = layd
			resp += 'new layer_dims = ' + str(self.layer_dims) + '\n'
			self.dinit = False # new dimensions require re-init
			self.activations = [self.activations[0]] * len(self.layer_dims)

		# iterations
		if ('iterations' in data['neuralnet']['hyper_params']):
			iters = data['neuralnet']['hyper_params']['iterations']
			if ((not isinstance(iters, int)) or (iters <= 0)):
				return Response('Error: invalid value for iterations. ' + str(iters), 405)
			self.iterations = iters
			resp += 'new iterations = ' + str(self.iterations) + '\n'

		# max_dataset_size
		if('max_dataset_size' in data['neuralnet']['hyper_params']):
			dset = data['neuralnet']['hyper_params']['max_dataset_size']
			if ((not isinstance(dset, int)) or (dset <= 0)):
				return Response('Error: invalid value for max data set size. ' + str(dset), 406)
			self.max_dataset_size = dset
			#self.dataset = [] # trim current dataset?
			resp += 'new max_dataset_size = ' + str(self.max_dataset_size) + '\n'

		# activations
		if ('activations' in data['neuralnet']['hyper_params']):
			acts = data['neuralnet']['hyper_params']['activations']
			
			# FIXME: validate sizes and stuff.  needs work
			if ((len(acts) != 1) and (len(acts) != len(self.layer_dims))):
				return Response('Error: activations need to match up to layer_dims or just be 1 value', 407)
			#for (a in acts):
			#	if (a != 'tanh' or a != 'linear' or a != 'sigmoid' or a != 'relu')
			#		return Response('Error:  invalid activation provided ' + str(a), 400)

			if len(acts) == 1:
				self.activations = acts * len(self.layer_dims)
			else:
				self.activations = acts
			resp += 'new activations = ' + str(self.activations) + '\n'

		# learning_rate
		if('learning_rate' in data['neuralnet']['hyper_params']):
			lrate = data['neuralnet']['hyper_params']['learning_rate']
			if (lrate <= 0):
				return Response('Error: invalid value for iterations. ' + str(lrate), 408)
			self.learning_rate = lrate
			resp += 'new learning_rate = ' + str(lrate) + '\n'

		return Response('Success: neuralnet updated. ' + resp, 202)


	def get_neuralnet(self, guid=None):
		#if guid is None: # TODO: DB junk
		data = {"neuralnet":{"hyper_params":{"learning_rate":self.learning_rate, 
			"activations":self.activations, "max_dataset_size":self.max_dataset_size,
			"iterations":self.iterations, "layer_dims":self.layer_dims}}}
		return Response(json.dumps(data), 200, mimetype='application/json')

	def generate_test_data(self):
		# generates training data using sin(x)s.  
		X = np.random.rand(3, self.max_dataset_size)# * 0.01
		Y = np.sin(X * np.pi)
		return (X, Y)
