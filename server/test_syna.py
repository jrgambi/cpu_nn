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


import unittest
import requests
import json

class TestSyna(unittest.TestCase):
	
	url = "http://127.0.0.1:5002"

	def test_get_health(self):
		response = requests.get(self.url + '/health')
		self.assertEqual(response.status_code, 200)

	def test_get_dataset_success(self):
		response = requests.get(self.url + '/dataset')
		self.assertEqual(response.status_code, 200)
		self.assertTrue('dataset' in response.json())

	# def test_get_dataset_failure(self): # TODO: when database is added failures will also happen

	def test_get_neuralnet_success(self):
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		rj = response.json()
		self.assertTrue('neuralnet' in rj)
		# TODO: validate other values, len(activations) == len(layer_dims)
		# TODO: get parameters as well?

	# def test_get_neuralnet_failure(self): # TODO: for DB

	def test_put_neuralnet_success(self):
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		og_nn = response.json()

		# layer_dims
		payload = {'neuralnet':{'hyper_params':{'layer_dims':[3,5,5,3]}}} # payload to update layer dimentions
		response = requests.put(self.url + '/neuralnet', json=payload)
		self.assertEquals(response.status_code, 202)
		
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		jr = response.json()
		self.assertEqual(len(jr['neuralnet']['hyper_params']['layer_dims']), 4)

		# iterations
		payload = {'neuralnet':{'hyper_params':{'iterations':3}}}
		response = requests.put(self.url + '/neuralnet', json=payload)
		self.assertEquals(response.status_code, 202)
		
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		jr = response.json()
		self.assertEquals(jr['neuralnet']['hyper_params']['iterations'], 3)

		# max_dataset_size
		payload = {'neuralnet':{'hyper_params':{'max_dataset_size':30}}}
		response = requests.put(self.url + '/neuralnet', json=payload)
		self.assertEquals(response.status_code, 202)
		
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		jr = response.json()
		self.assertEquals(jr['neuralnet']['hyper_params']['max_dataset_size'], 30)

		# activations
		acts = ['tanh', 'tanh', 'tanh', 'tanh']
		payload = {'neuralnet':{'hyper_params':{'activations':acts}}}
		response = requests.put(self.url + '/neuralnet', json=payload)
		self.assertEquals(response.status_code, 202)
		
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		jr = response.json()
		self.assertEquals(jr['neuralnet']['hyper_params']['activations'], acts)

		act = ['tanh']
		payload = {'neuralnet':{'hyper_params':{'activations':act}}}
		response = requests.put(self.url + '/neuralnet', json=payload)
		self.assertEquals(response.status_code, 202)
		
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		jr = response.json()
		self.assertEquals(jr['neuralnet']['hyper_params']['activations'], acts)


		# learning_rate
		payload = {'neuralnet':{'hyper_params':{'learning_rate': 0.73}}}
		response = requests.put(self.url + '/neuralnet', json=payload)
		self.assertEquals(response.status_code, 202)
		
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		jr = response.json()
		self.assertEquals(jr['neuralnet']['hyper_params']['learning_rate'], 0.73)

		# activations & layer_dims
		acts = ['tanh', 'tanh', 'tanh', 'tanh', 'sigmoid']
		payload = {'neuralnet':{'hyper_params':{'layer_dims':[3,5,5,5,3], 'activations':acts}}}
		response = requests.put(self.url + '/neuralnet', json=payload)
		self.assertEquals(response.status_code, 202)
		
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		jr = response.json()
		self.assertEquals(jr['neuralnet']['hyper_params']['activations'], acts)
		self.assertEqual(len(jr['neuralnet']['hyper_params']['layer_dims']), 5)


		# all together
		response = requests.put(self.url + '/neuralnet', json=og_nn)
		self.assertEquals(response.status_code, 202)

		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		self.assertEquals(response.json(), og_nn)


	def test_put_neuralnet_failure(self):
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		og_nn = response.json()

		# TODO:  all the fails
		
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		self.assertEquals(response.json(), og_nn)

	
	def test_post_dataset_success(self):
		response = requests.get(self.url + '/dataset')
		self.assertEqual(response.status_code, 200)
		og_dataset = response.json # empty for now at least...

		# TODO: layer_dims needs input and output sizes set [3, ... , 3]
		# TODO: this ^ part move to setup

		# post short dataset
		dataset = {'dataset':{'force_new':True, 'inputs':[[10,20,30],[40,50,60]], 'outputs':[[60,50,40], [30,20,10]]}}
		processed_dataset = {'dataset':{'inputs':[[10,20,30],[40,50,60]], 'outputs':[[60,50,40], [30,20,10]]}}

		response = requests.post(self.url + '/dataset', json=dataset)
		self.assertEqual(response.status_code, 200)
		# get dataset and assert
		response = requests.get(self.url + '/dataset')
		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json(), processed_dataset)

		# post short with force new
		dataset = {'dataset':{'force_new':True, 'inputs':[[1,2,3],[4,5,6]], 'outputs':[[6,5,4],[3,2,1]]}} # process inputs?
		processed_dataset = {'dataset':{'inputs':[[1,2,3],[4,5,6]], 'outputs':[[6,5,4],[3,2,1]]}}
		response = requests.post(self.url + '/dataset', json=dataset)
		self.assertEqual(response.status_code, 200)
		# get dataset and assert
		response = requests.get(self.url + '/dataset')
		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json(), processed_dataset)

		# post some more to append
		# get dataset and assert append and all is there
		dataset = {'dataset':{'inputs':[[1,2,3],[4,5,6]], 'outputs':[[6,5,4],[3,2,1]]}}
		processed_dataset = {'dataset':{'inputs':[[1,2,3],[4,5,6],[1,2,3],[4,5,6]], 'outputs':[[6,5,4],[3,2,1],[6,5,4],[3,2,1]]}}
		response = requests.post(self.url + '/dataset', json=dataset)
		self.assertEqual(response.status_code, 200)
		# get dataset and assert
		response = requests.get(self.url + '/dataset')
		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json(), processed_dataset)

		# post max, set max value first
		# get and assert

		# restore dataset

	def test_post_dataset_failure(self):
		response = requests.get(self.url + '/dataset')
		self.assertEqual(response.status_code, 200)
		og_dataset = response.json

		# TODO: set layer_dims[3, ... , 3]

		# post invalid values and assert errors
		# invalid request
		req = {'data':'data'}
		response = requests.post(self.url + '/dataset', json=req)
		self.assertEqual(response.status_code, 400)

		# invalid data sizes
		req = {'dataset':{'inputs':[[1,2]], 'outputs':[[2,1]]}} 
		response = requests.post(self.url + '/dataset', json=req)
		self.assertEqual(response.status_code, 400)

		# invalid data 
		req = {'dataset':{'inputs':[[1,2,3]], 'outputs':[[3,2,1],[3,2,1]]}} 
		response = requests.post(self.url + '/dataset', json=req)
		self.assertEqual(response.status_code, 400)

		# TODO: define more fails
