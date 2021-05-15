# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:12:45 2021

@author: Tianying Chu
"""
import pandas as pd
import numpy as np
import os

def readFiles(base_path):
    heart_disease_train = pd.read_csv(base_path + '/Data/Merged/Heart_merged_800.csv', index_col=0)
    stroke_train = pd.read_csv(base_path + '/Data/Merged/Stroke_merged_800.csv', index_col=0)
    test = pd.read_csv(base_path + '/Data/Merged/Test_merged_800.csv', index_col=0)
    return heart_disease_train, stroke_train, test

def imputeMean(df, vars_source):
    # Calculate state average
    interested_vars = list(df.columns[df.columns.str.endswith('_' + vars_source)])
    
    # Create inputing indicators
    indicators = df[interested_vars].isnull().astype(int).add_suffix('_ind')
        
    # Impute with state means
    for var in interested_vars:
        for year in df['year'].unique():
            df.loc[(df['state']=="District of Columbia") & (df['year'] == year), [var]] = \
                df[(df['state']=="Maryland") & (df['year'] == year)][var].mean() 
        df[var] = df.groupby(['state','year'])[var].transform(lambda group: group.fillna(group.mean()))
    
    filled = pd.concat([df, indicators], axis=1)
    return filled

def imputeRecentYear(df, vars_source):
    # Drop 3 disease columns with most missing data
    cols = df.columns[df.columns.str.contains('(hiv/aids)|(autism_spectrum_disorders)|(hepatitis_\(chronic_viral_b_&_c\))', regex=True)]
    df = df.drop(columns=cols)
    
    interested_vars = list(df.columns[df.columns.str.endswith('_' + vars_source)])
    
    # First impute with previous year numbers
    filled_previous = df[['fips', 'year'] + interested_vars].groupby(['fips']).fillna(method='ffill')  
    filled_previous = pd.concat([df['fips'], filled_previous], axis=1)
    
    # Then impute with next year numbers
    filled_next = filled_previous.groupby(['fips']).fillna(method='bfill')  
    filled_next = pd.concat([df['state'], filled_next], axis=1)
    
    # Finally impute with state average
    filled = (filled_next
              .fillna(filled_next
                      .groupby(['state', 'year']).transform('mean'))).drop(columns=['state', 'year'])
    
    # Create inputing indicators
    indicators = df[interested_vars].isnull().astype(int).add_suffix('_ind')
    filled = pd.concat([filled, indicators], axis=1)
    
    filled = pd.concat([df.drop(columns=list(interested_vars)), filled], axis=1)
    return filled

def imputeRemaining(df):
    interested_vars = ['year']
    for vars_source in ['aqi', 'county', 'state', 'prevalence', 'spending']:
        interested_vars += list(df.columns[df.columns.str.endswith('_' + vars_source)])
    
    # Impute with yearly average
    filled = (df[interested_vars]
              .fillna(df[interested_vars]
                      .groupby(['year']).transform('mean')))
    interested_vars.remove('year')
    filled = pd.concat([df.drop(columns=list(interested_vars)), filled], axis=1)
    return filled

def imputeAll(df):
    # Impute state mean for data from AQI & ACS
    df = imputeMean(df, 'aqi')
    df = imputeMean(df, 'county')
    df = imputeMean(df, 'state')
    df = imputeRecentYear(df, 'prevalence')
    df = imputeRecentYear(df, 'spending')
    df = imputeRemaining(df)
    return df

def imputer(train_feature_years, predict_feature_year, base_path):
    # Read all three files
    heart_disease_train, stroke_train, test = readFiles(base_path)
    
    # Impute the three files
    filled_heart_disease = imputeAll(heart_disease_train)
    filled_stroke = imputeAll(stroke_train)
    filled_test = imputeAll(test)
    
    # Save the imputed files
    if not os.path.exists(os.path.join(base_path, 'Data/Filled')):
        os.makedirs(os.path.join(base_path, 'Data/Filled'))
    filled_heart_disease.to_csv(base_path + '/Data/Filled/Heart_filled_800.csv')
    filled_stroke.to_csv(base_path + '/Data/Filled/Stroke_filled_800.csv')
    filled_test.to_csv(base_path + '/Data/Filled/Test_filled_800.csv')
    return