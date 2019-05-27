import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def logRegTrain(request):
    df = pd.DataFrame(request)
    categoricals = []
    for col, col_type in df.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df[col].fillna(0, inplace=True)

    df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)

    dependent_var = 'Prediction'

    x = df_ohe[df_ohe.columns.difference([dependent_var])]
    y = df_ohe[dependent_var]
    accuracy = trainHelper(x,y)


    model_columns = list(x.columns)
    joblib.dump(model_columns, 'LogisticRegression/LogRegModel_columns.pkl')
    print("Models columns dumped")

    return accuracy

def trainHelper(x,y):
    X = x.values
    Y = y.values
    m = X.shape[0]
    shuffle = np.random.permutation(np.arange(m))
    X = X[shuffle]
    Y = Y[shuffle]
    Xtrain, Ytrain = X[:int(m*0.6)], Y[:int(m*0.6)]
    Xtest, Ytest = X[int(m*0.6):], Y[int(m*0.6):]

    lr = LogisticRegression()
    lr.fit(Xtrain, Ytrain)
    predictions = lr.predict(Xtest)
    accuracy = np.mean(predictions == Ytest)

    joblib.dump(lr, 'LogisticRegression/LogRegModel.pkl')
    print("Model dumped")

    return accuracy

