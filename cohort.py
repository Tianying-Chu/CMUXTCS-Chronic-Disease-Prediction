# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:01:34 2021

@author: Tianying Chu
"""
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def readFiles(base_path):
    heart_disease_train = pd.read_csv(base_path + '/Data/Transformed/Heart_transformed_800.csv', index_col=0)
    stroke_train = pd.read_csv(base_path + '/Data/Transformed/Stroke_transformed_800.csv', index_col=0)
    return heart_disease_train, stroke_train

def getCohortRF(df, train_feature_years):
    cohort_train_years = train_feature_years[:-3]
    cohort_val_year = train_feature_years[-1]
    cohort_train = df[df['year'].isin(cohort_train_years)]
    cohort_val = df[df['year'] == int(cohort_val_year)]
    cohort_train = cohort_train.drop(columns=['year', 'fips'])
    cohort_val = cohort_val.drop(columns=['year', 'fips'])
    return cohort_train, cohort_val

def getCohortNN(df, train_feature_years, scaler):
    indFeat = 3
    cohort_train_years = train_feature_years[:-3]
    cohort_val_year = train_feature_years[-1]
    cohort_train_df = df[df['year'].isin(cohort_train_years)]
    cohort_val_df = df[df['year'] == int(cohort_val_year)]
    xTrain = cohort_train_df[cohort_train_df.columns[indFeat:]]
    xTrain = torch.from_numpy(np.array(xTrain))
    yTrain = torch.from_numpy(np.array(cohort_train_df.value))/scaler
    xValid = cohort_val_df[cohort_val_df.columns[indFeat:]]
    xValid = torch.from_numpy(np.array(xValid))
    yValid = torch.from_numpy(np.array(cohort_val_df.value))/scaler
    return (xTrain, yTrain), (xValid, yValid)

def helperRNN(year, DF, n, scaler):
    indFeat = 3
    nFeature = len(list(DF.columns)[indFeat:])
    x = np.zeros((len(year), n, nFeature))
    y = np.zeros((len(year), n))
    for i in range(len(year)):
        temp = DF[DF.year==year[i]]
        temp = temp.drop_duplicates(subset='fips', keep='first')
        temp = temp.sort_values(['fips'])
        x[i,:,:] = StandardScaler().fit_transform(temp[temp.columns[indFeat:]])
        y[i] = temp.value
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)/scaler
    return x, y

def getCohortRNN(data, train_feature_years, scaler):
    yearTrain = list(map(int, train_feature_years[:-3]))
    yearTest = list(map(int, train_feature_years))
    # Get a common set of counties
    fipsUni = data[data.year==yearTrain[0]].fips
    for yr in list(np.unique(yearTrain+yearTest)):
        fipsUni = np.intersect1d(fipsUni, data[data.year==yr].fips)
    n = len(fipsUni)
    fips = np.sort(fipsUni)
    data = data[data['fips'].isin(fipsUni)]
    trainDF = data[(data['year'].isin(yearTrain))]
    testDF = data[data['year'].isin(yearTest)]
    xTrain, yTrain = helperRNN(yearTrain, trainDF, n, scaler)
    xValid, yValid = helperRNN(yearTest, testDF, n, scaler)  
    return (xTrain, yTrain), (xValid, yValid), fips

def cohort(model_name, train_feature_years, predict_feature_year, base_path):
    heart_disease_train, stroke_train = readFiles(base_path)
    if model_name in ['RF', 'KNN', 'LR']:
        heart_disease_cohorts = getCohortRF(heart_disease_train, train_feature_years)
        stroke_cohorts = getCohortRF(stroke_train, train_feature_years)
    if model_name in ['NN']:
        heart_disease_cohorts = getCohortNN(heart_disease_train, train_feature_years, 100)
        stroke_cohorts = getCohortNN(stroke_train, train_feature_years, 20)
    if model_name in ['RNN']:
        heart_disease_cohorts = getCohortRNN(heart_disease_train, train_feature_years, 100)
        stroke_cohorts = getCohortRNN(stroke_train, train_feature_years, 20)
    
    return heart_disease_cohorts, stroke_cohorts
