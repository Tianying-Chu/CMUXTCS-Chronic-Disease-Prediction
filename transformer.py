# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:27:50 2021

@author: Tianying Chu
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def readFiles(base_path):
    heart_disease_train = pd.read_csv(base_path + '/Data/Filled/Heart_filled_800.csv', index_col=0)
    stroke_train = pd.read_csv(base_path + '/Data/Filled/Stroke_filled_800.csv', index_col=0)
    test = pd.read_csv(base_path + '/Data/Filled/Test_filled_800.csv', index_col=0)
    return heart_disease_train, stroke_train, test

def allZeroIndicators(df):
    # Get the indicator columns with all zeros
    ind_cols = df.columns[df.columns.str.endswith('_ind')]
    if_all_zero = df[ind_cols].sum(axis=0) == 0
    return ind_cols[if_all_zero]

def removeAllZero(heart_disease_train, stroke_train, test):
    # Find all zeros indicator columns in all three files
    heart_disease_zero = allZeroIndicators(heart_disease_train)
    stroke_zero = allZeroIndicators(stroke_train)
    test_zero = allZeroIndicators(test)
    all_zero = set(heart_disease_zero).intersection(set(stroke_zero),
                                                    set(test_zero))
    
    # Drop all zeros indicator columns
    heart_disease_train.drop(columns=all_zero, inplace=True)
    stroke_train.drop(columns=all_zero, inplace=True)
    test.drop(columns=all_zero, inplace=True)
    return heart_disease_train, stroke_train, test

def getCol(df):
    # Select interested columns
    aqi_cols = list(df.columns[df.columns.str.endswith('_aqi')])
    county_cols = list(df.columns[df.columns.str.endswith('_county')])
    state_cols = list(df.columns[df.columns.str.endswith('_state')])
    prevalence_cols = list(df.columns[df.columns.str.endswith('_prevalence')])
    spending_cols = list(df.columns[df.columns.str.endswith('_spending')])
    cols = aqi_cols + county_cols + state_cols + prevalence_cols + spending_cols
    return cols

def scalerTrain(df):
    cols = getCol(df)
    # Standardize the columns
    stdscaler = StandardScaler()
    stdscaler.fit(df.loc[:, cols])
    scaled_np = stdscaler.transform(df.loc[:, cols])
    scaled_df = pd.DataFrame(scaled_np, columns=cols)    
    scaled_df = pd.concat([df.drop(columns=cols), scaled_df], axis=1)
    return scaled_df, stdscaler

def scaler(df, stdscaler):
    cols = getCol(df)
    scaled_np = stdscaler.transform(df.loc[:, cols])
    scaled_df = pd.DataFrame(scaled_np, columns=cols)    
    scaled_df = pd.concat([df.drop(columns=cols), scaled_df], axis=1)
    return scaled_df

def scaleFeatures(heart_disease_train, stroke_train, test):
    heart_disease_train, heart_scaler = scalerTrain(heart_disease_train)
    stroke_train, stroke_scaler = scalerTrain(stroke_train)
    test = scaler(test, heart_scaler)
    return heart_disease_train, stroke_train, test

def oheState(heart_disease_train, stroke_train, test):
    # Fit one hot encoding columns
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe.fit(heart_disease_train[['state']])
    cols = [name.replace('x0', 'state') for name in ohe.get_feature_names()]
    
    # Transform all three files
    heart_disease_ohe_np = ohe.transform(heart_disease_train[['state']])
    heart_disease_ohe_df = pd.DataFrame(heart_disease_ohe_np, columns=cols)    
    heart_disease_ohe_df = pd.concat([heart_disease_train.drop(columns=['state']), heart_disease_ohe_df], axis=1)
    
    stroke_ohe_np = ohe.transform(stroke_train[['state']])
    stroke_ohe_df = pd.DataFrame(stroke_ohe_np, columns=cols)    
    stroke_ohe_df = pd.concat([stroke_train.drop(columns=['state']), stroke_ohe_df], axis=1)
    
    test_ohe_np = ohe.transform(test[['state']])
    test_ohe_df = pd.DataFrame(test_ohe_np, columns=cols)    
    test_ohe_df = pd.concat([test.drop(columns=['state']), test_ohe_df], axis=1)
    
    return heart_disease_ohe_df, stroke_ohe_df, test_ohe_df

def removeNanHospitalization(df):
    df.replace('Insufficient Data',np.nan,inplace=True)
    df.value = df.value.astype(float)
    df = df[df['value'].isnull()!=True]
    return df

def prepareFeatures(heart_disease_train, stroke_train, test):
    # Remove rows missing outcomes
    heart_disease_train = removeNanHospitalization(heart_disease_train)
    stroke_train = removeNanHospitalization(stroke_train)
    
    # Feature engineering
    heart_disease_train, stroke_train, test = removeAllZero(heart_disease_train, stroke_train, test)
    heart_disease_train, stroke_train, test = scaleFeatures(heart_disease_train, stroke_train, test)
    
    # Drop unnecessary columns
    heart_disease_train = heart_disease_train.drop(columns=['state', 'geo_id', 'state_abbr',
                                                            'county',  'predict_year', 'range', 'year.1'])
    stroke_train = stroke_train.drop(columns=['state', 'geo_id', 'state_abbr', 
                                              'county',  'predict_year', 'range', 'year.1'])
    test = test.drop(columns=['state', 'geo_id', 'state_abbr', 'county', 'year.1'])
    return heart_disease_train, stroke_train, test

def transformer(train_feature_years, predict_feature_year, base_path):
    heart_disease_train, stroke_train, test = readFiles(base_path)
    heart_disease_train, stroke_train, test = prepareFeatures(heart_disease_train, stroke_train, test)
    
    # Save the transformed files
    if not os.path.exists(os.path.join(base_path, 'Data/Transformed')):
        os.makedirs(os.path.join(base_path, 'Data/Transformed'))
    heart_disease_train.to_csv(base_path + '/Data/Transformed/Heart_transformed_800.csv')
    stroke_train.to_csv(base_path + '/Data/Transformed/Stroke_transformed_800.csv')
    test.to_csv(base_path + '/Data/Transformed/Test_transformed_800.csv')
    
    return 