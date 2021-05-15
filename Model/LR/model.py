from sklearn.linear_model import Ridge
import joblib

def learnLR(X_train, y_train, load_model, model_type):
    if load_model == False:
        regressor = Ridge()
        regressor.fit(X_train, y_train)
    if load_model == True:
        regressor = joblib.load('Model/LR/{}'.format(model_type))
    return regressor