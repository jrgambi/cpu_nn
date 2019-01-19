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
		# TODO: empty json?

	# def test_get_dataset_failure(self): # TODO: when database is added failures will also happen

	def test_get_neuralnet_success(self):
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		rj = response.json()
		self.assertTrue('neuralnet' in rj)

	# def test_get_neuralnet_failure(self): # TODO: for DB

	def test_put_neuralnet_success(self):
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		og_nn = response.json()

		payload = {'neuralnet':{'hyper_params':{'layer_dims':[3,5,5,3]}}} # payload to update layer dimentions
		response = requests.put(self.url + '/neuralnet', json=payload)
		self.assertEquals(response.status_code, 202)
		
		response = requests.get(self.url + '/neuralnet')
		self.assertEqual(response.status_code, 200)
		jr = response.json()
		self.assertEqual(len(jr['neuralnet']['hyper_params']['layer_dims']), 4)

		# TODO: more success

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

		# TODO: following
		# post short dataset
		# get dataset and assert
		
		# post some more to append
		# get dataset and assert append and all is there

		# post max
		# get and assert

		# restore dataset

	def test_post_dataset_failure(self):
		response = requests.get(self.url + '/dataset')
		self.assertEqual(response.status_code, 200)
		og_dataset = response.json

		# post invalid values and assert errors
		# TODO: add more detailed notes 
