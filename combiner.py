# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 21:01:46 2021

@author: Tianying Chu
"""

import pandas as pd
import numpy as np
import os

def readFile(path):
    df = pd.read_csv(path, index_col=0, dtype={'fips': str})
    df['fips'] = df['fips'].str.pad(5, side='left', fillchar='0')
    return df

def readHospitalization(path):
    Hospitalization = readFile(path)
    Hospitalization.rename(columns={'year': 'predict_year',
                                    'County': 'county'}, inplace=True)
    Hospitalization['year'] = (
        Hospitalization['predict_year'].str[:4].astype('int')
        .apply(lambda x: x-1)
    )
    return Hospitalization

def readAQI(path):
    AQI = readFile(path)
    AQI.rename(columns={col: col.replace(' ', '_').lower() 
                        for col in AQI.columns.values}, 
               inplace=True)
    return AQI

def readACS(path):
    ACS = readFile(path)
    ACS['state'] = ACS['state'].str.strip()
    ACS.reset_index(drop=True, inplace=True)
    return ACS

def readCMS(path):
    CMS = pd.read_csv(path, index_col=0, dtype={'FIPS': str}, na_values='* ')
    CMS.rename(columns={col: col.strip().replace(' ', '_').lower() 
                        for col in CMS.columns.values},
               inplace=True)
    CMS['fips'] = CMS['fips'].str.strip()
    CMS['fips'] = CMS['fips'].str.pad(5, side='left', fillchar='0')
    return CMS

def yearFilter(data, years):
    return data[data['year'].isin(years)]

def reorderColumns(df, columns_order):
    cols = list(df)
    for i in range(len(columns_order)):
        cols.insert(i, cols.pop(cols.index(columns_order[i])))
    return df.loc[:,cols]

def mergeTrain(train_feature_years, hospitalization, AQI, prev_8_AQI, prev_16_AQI, ACS_county, ACS_state, prevalence, spending):
    # Filter training year
    AQI_train = yearFilter(AQI, train_feature_years)
    prev_8_AQI_train = yearFilter(prev_8_AQI, train_feature_years)
    ACS_county_train = yearFilter(ACS_county, train_feature_years)
    ACS_state_train = yearFilter(ACS_state, train_feature_years)
    prevalence_train = yearFilter(prevalence, train_feature_years)
    spending_train = yearFilter(spending, train_feature_years)
    
    merged = (
        hospitalization.drop(columns=['county'])
        .merge(ACS_county_train, how='inner', 
               left_on=['fips', 'year'], right_on=['fips', 'year'])
        .merge(ACS_state_train.drop(columns=['fips']), how='left', 
               left_on=['state', 'year'], right_on=['state', 'year'], suffixes=['_county', '_state'])
        .merge(AQI_train.drop(columns=['geo_id', 'county', 'state_abbr', 'state']).add_suffix('_aqi'), 
               how='left', 
               left_on=['fips', 'year'], right_on=['fips_aqi', 'year_aqi'])
        .drop(columns=['fips_aqi', 'year_aqi'])
        .merge(prev_8_AQI_train.drop(columns=['geo_id', 'county', 'state_abbr', 'state', 'prev_8_year']).add_suffix('_prev_8_aqi'), 
               how='left', 
               left_on=['fips', 'year'], right_on=['fips_prev_8_aqi', 'year_prev_8_aqi'])
        .drop(columns=['fips_prev_8_aqi', 'year_prev_8_aqi'])
        .merge(prevalence_train.drop(columns=['state', 'county']), how='left',
               left_on=['fips', 'year'], right_on=['fips', 'year'])
        .merge(spending_train.drop(columns=['state', 'county']), how='left',
               left_on=['fips', 'year'], right_on=['fips', 'year'], suffixes=['_prevalence', '_spending'])
        )
    
    # Reorder columns
    columns_order = ['fips', 'geo_id', 'state', 'state_abbr', 'county',
                     'year', 'predict_year', 'value', 'range']
    merged = reorderColumns(merged, columns_order)
    #print(merged.columns)
    return merged

def mergeTest(predict_feature_year_lst, AQI, prev_8_AQI, prev_16_AQI, ACS_county, ACS_state, prevalence, spending):
    # Filter prediction year
    AQI_prediction = yearFilter(AQI, predict_feature_year_lst)
    prev_8_AQI_prediction = yearFilter(prev_8_AQI, predict_feature_year_lst)
    ACS_county_prediction = yearFilter(ACS_county, predict_feature_year_lst)
    ACS_state_prediction = yearFilter(ACS_state, predict_feature_year_lst)
    prevalence_prediction = yearFilter(prevalence, predict_feature_year_lst)
    spending_prediction = yearFilter(spending, predict_feature_year_lst)
    
    merged = (
        ACS_county_prediction
        .merge(ACS_state_prediction.drop(columns=['fips']), how='left', 
               left_on=['state', 'year'], right_on=['state', 'year'], suffixes=['_county', '_state'])
        .merge(AQI_prediction.drop(columns=['county', 'state']).add_suffix('_aqi'), 
               how='left', 
               left_on=['fips', 'year'], right_on=['fips_aqi', 'year_aqi'])
        .drop(columns=['fips_aqi', 'year_aqi'])
        .rename(columns={'geo_id_aqi': 'geo_id', 'state_abbr_aqi': 'state_abbr'})
        .merge(prev_8_AQI_prediction.drop(columns=['geo_id', 'county', 'state_abbr', 'state', 'prev_8_year']).add_suffix('_prev_8_aqi'), 
               how='left', 
               left_on=['fips', 'year'], right_on=['fips_prev_8_aqi', 'year_prev_8_aqi'])
        .drop(columns=['fips_prev_8_aqi', 'year_prev_8_aqi'])
        .merge(prevalence_prediction.drop(columns=['state', 'county']), how='left',
               left_on=['fips', 'year'], right_on=['fips', 'year'])
        .merge(spending_prediction.drop(columns=['state', 'county']), how='left',
               left_on=['fips', 'year'], right_on=['fips', 'year'], suffixes=['_prevalence', '_spending'])
        )
    
    # Reorder columns
    columns_order = ['fips', 'geo_id', 'state', 'state_abbr', 'county', 'year']
    merged = reorderColumns(merged, columns_order)
    return merged

def combiner(train_feature_years, predict_feature_year_lst, base_path):
    # Read all data sources
    heart_disease = readHospitalization(base_path + '/Data/Data_with_FIPS/Heart_Disease_FIPS.csv')
    stroke = readHospitalization(base_path + '/Data/Data_with_FIPS/Stroke_FIPS.csv')
    AQI = readAQI(base_path + '/Data/Data_with_FIPS/AQI_FIPS.csv')
    prev_8_AQI = readAQI(base_path + '/Data/Data_with_FIPS/prev_8_AQI_FIPS.csv')
    prev_16_AQI = readAQI(base_path + '/Data/Data_with_FIPS/prev_16_AQI_FIPS.csv')
    ACS_county = readACS(base_path + '/Data/Data_with_FIPS/ACS_County_FIPS.csv')
    ACS_state = readACS(base_path + '/Data/Data_with_FIPS/ACS_State_FIPS.csv')
    prevalence = readCMS(base_path + '/Data/Data_with_FIPS/Prevalence_FIPS.csv')
    spending = readCMS(base_path + '/Data/Data_with_FIPS/Spending_FIPS.csv')
    
    # Combine training sets and test set
    heart_disease_train = mergeTrain(train_feature_years, heart_disease, AQI, prev_8_AQI, prev_16_AQI,
                                     ACS_county, ACS_state, prevalence, spending)
    stroke_train = mergeTrain(train_feature_years, stroke, AQI, prev_8_AQI, prev_16_AQI,
                                     ACS_county, ACS_state, prevalence, spending)
    test = mergeTest(predict_feature_year_lst, AQI, prev_8_AQI, prev_16_AQI, ACS_county, ACS_state, 
                     prevalence, spending)
    
    # Save all three files
    if not os.path.exists(os.path.join(base_path, 'Data/Merged')):
        os.makedirs(os.path.join(base_path, 'Data/Merged'))
    heart_disease_train.to_csv(base_path + '/Data/Merged/Heart_merged_800.csv')
    stroke_train.to_csv(base_path + '/Data/Merged/Stroke_merged_800.csv')
    test.to_csv(base_path + '/Data/Merged/Test_merged_800.csv')
    return
