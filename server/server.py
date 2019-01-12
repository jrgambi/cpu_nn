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
		response = be.process_data(data)
		return str(response)
	else:
		return str(be.dataset)

@app.route('/neuralnet', methods=['GET', 'PUT', 'POST'])
def neuralnet():
	if request.method == 'PUT':
		# modify neural network
		print '/neuralnet PUT received'

	if request.method == 'POST':
		#start/stop # TODO: base action on POST data
		action = request.json['action']
		# actions: start, stop, continue
		if action == 'start':
			be.train = True
			be.start_training()
		elif action == 'stop':
			be.train = False
		else:
			# no valid action posted
			return 'inaction is :(\n'

	return 'nn\n'

if __name__ == '__main__':
	app.run(port='5002')
