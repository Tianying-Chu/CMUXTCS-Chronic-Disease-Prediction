# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:58:45 2021

@author: Tianying Chu
"""
import pandas as pd
from Model.RF.model import learnRF
from Model.NN.model import learnNN
from Model.KNN.model import learnKNN
from Model.RNN.model import learnRNN
from Model.LR.model import learnLR
import torch
import joblib

def getFeaturesLabel(model_name, df):
    if model_name in ['NN', 'RNN']:
        xTrain = df[0]
        yTrain = df[1]
    if model_name in ['RF', 'KNN', 'LR']:
        yTrain = df['value']
        xTrain = df.drop(columns=['value'])
    return xTrain, yTrain

def train(model_name, load_model, cohorts, model_type, base_path):
    if model_name in ['NN']:
        if model_type == 'heart':
            train_set = cohorts[0][0]
            epoch = 10000
            lr = 0.005
            momentum = 0.9
            lambda_l2 = 0.001
        if model_type == 'stroke':
            train_set = cohorts[1][0]
            epoch = 5000
            lr = 0.005
            momentum = 0.9
            lambda_l2 = 0.001
        xTrain, yTrain = getFeaturesLabel(model_name, train_set)
        model = learnNN(lambda_l2, epoch, lr, momentum, xTrain, yTrain, load_model, model_type)
        torch.save(model.state_dict(), 'Result/{}/{}'.format(model_name, model_type))

    if model_name in ['RF', 'KNN', 'LR']:
        if model_type == 'heart':
            train_set = cohorts[0][0]
        if model_type == 'stroke':
            train_set = cohorts[1][0]
        xTrain, yTrain = getFeaturesLabel(model_name, train_set)
        if model_name == 'RF':
            model = learnRF(xTrain, yTrain, load_model, model_type)
        if model_name == 'KNN':
            model = learnKNN(xTrain, yTrain, load_model, model_type)
        if model_name == 'LR':
            model = learnLR(xTrain, yTrain, load_model, model_type)
        joblib.dump(model, 'Result/{}/{}'.format(model_name, model_type))

    if model_name in ['RNN']:
        if model_type == 'heart':
            train_set = cohorts[0][0]
            nHidden = 20
            epoch = 10000
            lr = 0.01
            momentum = 0.9
            lambda_l2 = 0.001
        if model_type == 'stroke':
            train_set = cohorts[1][0]
            nHidden = 20
            epoch = 10000
            lr = 0.01
            momentum = 0.9
            lambda_l2 = 0.001
        xTrain, yTrain = getFeaturesLabel(model_name, train_set)
        model = learnRNN(lambda_l2, nHidden, epoch, lr, momentum, xTrain, yTrain, load_model, model_type)
        torch.save(model.state_dict(), 'Result/{}/{}'.format(model_name, model_type))   
    return model
