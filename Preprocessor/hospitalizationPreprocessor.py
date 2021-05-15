# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:17:42 2021

@author: Tianying Chu
"""

from Preprocessor.preprocessor import countyNamePreprocessor, readCountyFips, mergeYears

def readHospitalization(Hospitalization_paths):
    # Read and combine Hospitalization data
    Hospitalization = mergeYears(Hospitalization_paths)
    Hospitalization = countyNamePreprocessor(Hospitalization)
    return Hospitalization

def mergyHospitalizationFips(Hospitalization, county):
    # Merge Hospitalization data with FIPS code
    Hospitalization_FIPS = (
        county.merge(Hospitalization, left_on=['state_abbr', 'County'], right_on=['state', 'County'])
    )
    Hospitalization_FIPS.drop(['state', 'state_name', 'long_name'], axis=1, inplace=True)
    Hospitalization_FIPS.drop_duplicates(['fips', 'year'], keep='first', inplace=True)
    return Hospitalization_FIPS

def hospitalizationPreprocessor(train_label_years, base_path):
    Heart_Disease_paths = [base_path+'/Data/Heart_Disease/All Heart Disease'+year+'.csv' 
                           for year in train_label_years]
    Stroke_paths = [base_path+'/Data/Stroke/All Stroke'+year+'.csv' 
                           for year in train_label_years]
    
    Heart_Disease = readHospitalization(Heart_Disease_paths)
    Stroke = readHospitalization(Stroke_paths)
    county = readCountyFips(base_path)
    
    Heart_Disease_FIPS = mergyHospitalizationFips(Heart_Disease, county)
    Stroke_FIPS = mergyHospitalizationFips(Stroke, county)
    
    Heart_Disease_FIPS.to_csv(base_path + '/Data/Data_with_FIPS/Heart_Disease_FIPS.csv')
    Stroke_FIPS.to_csv(base_path + '/Data/Data_with_FIPS/Stroke_FIPS.csv')
    return
