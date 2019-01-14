from flask import Flask, request, Response
from flask_restful import Resource, Api
from syna import *

app = Flask(__name__)
api = Api(app)

be = Backend()

@app.route('/health', methods=['GET'])
def health():
	return Response('OK', 200) # 

@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
	if request.method == 'POST':
		#update dataset
		data = request.json
		print '/dataset POST received: ' + str(data)
		response = be.process_dataset(data)
		return response
	else:
		return Response(status=200)

@app.route('/neuralnet', methods=['GET', 'PUT', 'POST'])
def neuralnet():
	if request.method == 'PUT':
		# modify neural network
		data = request.json
		print '/neuralnet PUT received: ' + str(data)
		response = be.modify_neuralnet(data)
		return response

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
			return Response(status=200)

	return 'nn\n'

if __name__ == '__main__':
	app.run(port='5002')
