from flask import Flask, jsonify, request
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import os

def prediction(model, model_columns, request):
    try:
        query = pd.get_dummies(pd.DataFrame(request)) #Allowing for categorical variables to be encoded
        query = query.reindex(columns=model_columns, fill_value=0) # relabel with the encoded labels
        prediction = list(model.predict(query)) # make a prediction
        return {'prediction': str(prediction)}
    except:
        return {'trace': traceback.format_exc()}
