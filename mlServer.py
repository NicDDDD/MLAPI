from flask import Flask, jsonify, request
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import os
from LogisticRegression.logRegModel import logRegTrain
from Clustering.ClusterModel import clusterTrain 
from serverHelpers import *

app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to my Machine Learning API"

@app.route("/logReg/train", methods=['POST'])
def trainLog():
    accuracy = logRegTrain(request.json)
    return jsonify({"Accuracy": str(accuracy)})

@app.route("/cluster/train", methods=['POST'])
def trainCluster():
    accuracy = clusterTrain(request.json)
    return jsonify({"Accuracy": str(accuracy)})

@app.route('/logReg/predict', methods=['POST'])
def predictLog():
    if os.path.isfile('LogisticRegression/logRegModel.pkl'):
        lr = joblib.load('LogisticRegression/logRegModel.pkl')
        model_columns = joblib.load('LogisticRegression/LogRegModel_columns.pkl') # loading model info
        answer = prediction(lr, model_columns, request.json)
        return jsonify(answer)
    else:
        print('Train the model first')
        return ('No model here to use')

@app.route('/cluster/predict', methods=['POST'])
def predictClust():
    if os.path.isfile('Clustering/ClusterModel.pkl'):
        km = joblib.load('Clustering/ClusterModel.pkl')
        model_columns = joblib.load('Clustering/ClusterModel_columns.pkl') # loading model info
        answer = prediction(km, model_columns, request.json)
        return jsonify(answer)
    else:
        print('Train the model first')
        return ('No model here to use')

            

if __name__ == '__main__':
    try: 
        port = int(sys.argv[1])
    except:
        port = 5000
    app.run(port = port, debug=True)