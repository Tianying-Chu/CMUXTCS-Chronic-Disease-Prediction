from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import joblib

def learnKNN(X_train, y_train, load_model, model_type):
    if load_model == False:
        regressor = KNeighborsRegressor(n_neighbors=3, p=2, weights='distance')
        regressor.fit(X_train, y_train)
    if load_model == True:
        regressor = joblib.load('Model/KNN/{}'.format(model_type))
    return regressor