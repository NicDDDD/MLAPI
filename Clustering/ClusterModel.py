import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def clusterTrain(request):
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
    joblib.dump(model_columns, 'Clustering/ClusterModel_columns.pkl')
    print("Models columns dumped")

    return accuracy

def trainHelper(x,y):
    X = x.values
    Y = y.values
    m = X.shape[0]
    shuffle = np.random.permutation(np.arange(m))
    X = X[shuffle]
    Y = Y[shuffle]
    Xtrain = X[:int(m*0.6)]
    Xtest, Ytest = X[int(m*0.6):], Y[int(m*0.6):]

    Km = KMeans(n_clusters=2)
    Km.fit(Xtrain)
    predictions = Km.predict(Xtest)
    accuracy = np.mean(predictions == Ytest)

    joblib.dump(Km, 'Clustering/ClusterModel.pkl')
    print("Model dumped")

    return accuracy