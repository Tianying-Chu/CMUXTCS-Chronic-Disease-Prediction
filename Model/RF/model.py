from sklearn.ensemble import RandomForestRegressor
import joblib

def learnRF(X_train, y_train, load_model, model_type):
    if load_model == False:
        regressor = RandomForestRegressor(n_estimators=20, random_state=0)
        regressor.fit(X_train, y_train)
    if load_model == True:
        regressor = joblib.load('Model/RF/{}'.format(model_type))
    return regressor