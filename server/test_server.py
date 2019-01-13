import unittest
import requests

class TestServer(unittest.TestCase):
	
	url = "http://127.0.0.1:5002"

	def test_get_health(self):
		response = requests.get(self.url + '/health')
		self.assertEqual(response.status_code, 200)
