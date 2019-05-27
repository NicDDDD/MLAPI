# Machine Learning API

This project is intended to be a Flask server that preforms data analysis on data sent to certain endpoints

## How to Use and Information:

Make sure you have at least Python 3.6.2 and clone this repository. Once cloned, navigate to the mlServer.py directory and run the command 

```
python mlServer.py
```

Once this is done, you can use postman to send requests to the endpoints made so far. 

The endpoints that have been made are:
- '/' : The home endpoint
- '/logReg/train' : For training a logistic Regression model
- '/logReg/predict' : For sending new data for making a prediction from the trained model
- '/cluster/train' : Same as before but for KMeans
- '/cluster/predict' : Same but for KMeans predictions

The data needs to be sent in json format similar to what is shown below for the predict endpoints:

```JSON
[
    {"Age": 85, "Sex": "male", "Embarked": "S"},
    {"Age": 24, "Sex": "female", "Embarked": "C"},
    {"Age": 3, "Sex": "male", "Embarked": "C"},
    {"Age": 21, "Sex": "male", "Embarked": "S"}
]

```
The training endpoints need a superised component, the column that we are tryoing to predict:

```JSON
[
    {"Age": 85, "Sex": "male", "Embarked": "S", "Prediction": 0 },
    {"Age": 24, "Sex": "female", "Embarked": "C", "Prediction": 1 },
    {"Age": 3, "Sex": "male", "Embarked": "C", "Prediction": 0 },
    {"Age": 21, "Sex": "male", "Embarked": "S", "Prediction": 0 }
]

```

For the training endpoints, the accuracy of the model is reported. The predict endpoints give back a vector of which binary choice the model thinks is correct.

### This API was mainly tested data that can be accessed below and therefore needs further abstraction

Data can be found [here](http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv
)

