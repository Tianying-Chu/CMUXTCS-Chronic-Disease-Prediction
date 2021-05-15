# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 23:02:37 2021

@author: Tianying Chu
"""

import numpy as np
from train import getFeaturesLabel
from sklearn import metrics

def testMetrics(model, model_name, model_type, cohorts, base_path):
    if model_type == 'heart':
        train_set = cohorts[0][0]
        test_set = cohorts[0][1]
    if model_type == 'stroke':
        train_set = cohorts[1][0]
        test_set = cohorts[1][1]   
    xTrain, yTrain = getFeaturesLabel(model_name, train_set)
    xTest, yTest = getFeaturesLabel(model_name, test_set)
    if model_name in ['NN']:
        yTrainPredict = model(xTrain).reshape(-1).cpu().detach().numpy()
        yTestPredict = model(xTest).reshape(-1).cpu().detach().numpy()
        yTrain = yTrain.detach()
        yTest = yTest.detach()   
    if model_name in ['RF', 'KNN', 'LR']:
        yTrainPredict = model.predict(xTrain)
        yTestPredict = model.predict(xTest)
    if model_name in ['RNN']:
        yTrainPredict = model(xTrain)[:,-1].reshape(-1).cpu().detach().numpy()
        yTestPredict = model(xTest)[:,-1].reshape(-1).cpu().detach().numpy()
        yTrain = yTrain[-1].detach()
        yTest = yTest[-1].detach()
    trainR2 = metrics.r2_score(yTrain, yTrainPredict)
    testR2 = metrics.r2_score(yTest, yTestPredict)
    print('Train R2:', round(trainR2, 2))
    print('Test R2:', round(testR2, 2))
    return