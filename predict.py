# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 23:04:46 2021

@author: Tianying Chu
"""
import pandas as pd
import torch
import numpy as np
from train import getFeaturesLabel
from sklearn.preprocessing import StandardScaler
from cohort import helperRNN
from Model.RNN.model import nNet

def readFiles(base_path):
    heart_df = pd.read_csv(base_path + '/Data/Transformed/Heart_transformed_800.csv', index_col=0)
    stroke_df = pd.read_csv(base_path + '/Data/Transformed/Stroke_transformed_800.csv', index_col=0)
    predict_df = pd.read_csv(base_path + '/Data/Transformed/Test_transformed_800.csv', index_col=0)
    return heart_df, stroke_df, predict_df

def readFilesRaw(base_path):
    heart_df = pd.read_csv(base_path + '/Data/Filled/Heart_filled_800.csv', index_col=0)
    predict_df = pd.read_csv(base_path + '/Data/Filled/Test_filled_800.csv', index_col=0)
    feature_train_df = heart_df.drop(columns=['state_abbr', 'value', 'range', 'predict_year'])
    feature_predict_df = predict_df.drop(columns=['state_abbr'])
    return feature_train_df, feature_predict_df

def scaling(model_name, train_df, predict_df, scaler):
    if model_name in ['NN', 'RF', 'KNN', 'LR']:
        xTrain = train_df[train_df.columns[3:]]
        xPredict = predict_df[predict_df.columns[2:]]
        if model_name in ['NN']:
            xTrain = torch.from_numpy(np.array(xTrain))
            xPredict = torch.from_numpy(np.array(xPredict))
        data_out = (xTrain, xPredict)
    if model_name in ['RNN']:
        predict_df.insert(2, 'value', [-99] * len(predict_df))
        data = pd.concat([train_df, predict_df])
        year = np.sort(np.unique(data.year))
        fipsUni = data[data.year==year[0]].fips
        for yr in list(year):
            fipsUni = np.intersect1d(fipsUni, data[data.year==yr].fips)
        n = len(fipsUni)
        fips = np.sort(fipsUni)
        data = data[data['fips'].isin(fipsUni)]
        x, y = helperRNN(year, data, n, scaler)
        data_out = (x, y, fips, year)
    return data_out

def output(train_df, predict_df, base_path, model_name, model_type):
    train_df = train_df[['fips', 'year', 'value', 'value_hat']]
    predict_df = predict_df[['fips', 'year', 'value_hat']]
    feature_train_df, feature_predict_df = readFilesRaw(base_path)
    train_df = pd.merge(left=train_df, right=feature_train_df, on=['fips', 'year'], how='left')
    predict_df = pd.merge(left=predict_df, right=feature_predict_df, on=['fips', 'year'], how='left')
    train_df.to_csv(base_path + '/Result/{}/{}_prediction_label.csv'.format(model_name, model_type), index=False)
    predict_df.to_csv(base_path + '/Result/{}/{}_prediction_nolabel.csv'.format(model_name, model_type), index=False)
    return train_df, predict_df

def shapRNN(x, model, model_type, data, base_path):
    hidden = torch.tanh(torch.matmul(x[-2,:,:], model.rnn.weight_ih_l0.T) + model.rnn.bias_ih_l0)
    output = torch.sigmoid(torch.matmul(hidden, model.output.weight.T) + model.output.bias)
    xFinal = torch.cat((x[-1,:,:], hidden, output), dim=1)
    l0 = torch.cat((model.rnn.weight_ih_l0, model.rnn.weight_hh_l0, model.output.weight.T), dim=1)
    b0 = model.rnn.bias_ih_l0 + model.rnn.bias_hh_l0 + model.output.bias.T
    neuralNet = nNet(20, l0, model.output.weight, b0, model.output.bias).double()

    df = pd.DataFrame(xFinal.detach().numpy())
    hiddenName = ['Previous Hidden {}'.format(i) for i in range(20)]
    df.columns = list(data.columns)[3:]+hiddenName+['Previous Output']

    torch.save(neuralNet.state_dict(), 'Result/RNN/{}.shap'.format(model_type))
    df.to_csv('Result/RNN/{}.csv'.format(model_type))
    return 

def prediction(model, model_name, model_type, base_path):
    pd.options.mode.chained_assignment = None
    heart_df, stroke_df, predict_df = readFiles(base_path)
    if model_type == 'heart':
        scaler = 100
        data_out = scaling(model_name, heart_df, predict_df, scaler)
        train_df = heart_df.copy()       
    if model_type == 'stroke':
        scaler = 20
        data_out = scaling(model_name, stroke_df, predict_df, scaler)
        train_df = stroke_df.copy()
    if model_name in ['NN']:
        xTrain = data_out[0]
        xPredict = data_out[1]
        yTrainHat = model(xTrain).reshape(-1).cpu().detach().numpy() * scaler
        yPredictHat = model(xPredict).reshape(-1).cpu().detach().numpy() * scaler
        train_df['value_hat'] = yTrainHat
        predict_df['value_hat'] = yPredictHat  
    if model_name in ['RF', 'KNN', 'LR']:
        xTrain = data_out[0]
        xPredict = data_out[1]
        yTrainHat = model.predict(xTrain)
        yPredictHat = model.predict(xPredict)
        train_df['value_hat'] = yTrainHat
        predict_df['value_hat'] = yPredictHat
    if model_name in ['RNN']:
        x = data_out[0]
        y = data_out[1]
        fips = data_out[2]
        year = data_out[3]
        result_list = []
        for i in range(len(year)):
            yr = year[i]
            value = y[i] * scaler
            yHat = model(x[:i+1,:,:])[:,-1].reshape(-1).cpu().detach().numpy() * scaler
            result_list.append(pd.DataFrame({
                'fips': fips,
                'year': [yr]*len(fips),
                'value': value,
                'value_hat': yHat
            }))
        result = pd.concat(result_list)
        train_df = result[result.value!=-99]
        predict_df = result[result.value==-99]
        predict_df = predict_df.drop(columns=['value'])
        shapRNN(x, model, model_type, heart_df, base_path)
    train_df, predict_df = output(train_df, predict_df, base_path, model_name, model_type)
    return train_df, predict_df
