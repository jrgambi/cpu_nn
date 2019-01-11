from flask import Flask, request
from flask_restful import Resource, Api
from server_backend import *

app = Flask(__name__)
api = Api(app)

be = Backend()

@app.route('/health', methods=['GET'])
def health():
	return 'OK\n'

@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
	if request.method == 'POST':
		#update dataset
		data = request.json
		print '/dataset POST received ' + str(data['dataset'])
		if be.validate_data(data):
			print 'valid '
		else:
			print 'invalid '
	return 'data\n'

@app.route('/neuralnet', methods=['GET', 'PUT'])
def neuralnet():
	if request.method == 'PUT':
		# modify neural network
		print '/neuralnet PUT received'
	return 'nn\n'

if __name__ == '__main__':
	app.run(port='5002')
