# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:57:57 2021

@author: Tianying Chu
"""

from Preprocessor.preprocessor import countyNamePreprocessor, readCountyFips, mergeYears

def readAqi(AQI_paths):
    # Read and combine AQI data
    AQI = mergeYears(AQI_paths)
    AQI = countyNamePreprocessor(AQI)
    return AQI

def mergyAqiFips(AQI, county):
    # Merge AQI data with FIPS code
    AQI_FIPS = (
        county.merge(AQI, left_on=['state_name', 'County'], right_on=['State', 'County'])
    )
    AQI_FIPS.drop(['state_name', 'long_name'], axis=1, inplace=True)
    AQI_FIPS.drop_duplicates(['fips', 'Year'], keep='first', inplace=True)
    return AQI_FIPS

def dayToPercent(AQI_FIPS):
    # Convert columns about days to percentage
    AQI_FIPS['Good Days Pct'] = AQI_FIPS['Good Days'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Moderate Days Pct'] = AQI_FIPS['Moderate Days'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Unhealthy for Sensitive Groups Days Pct'] = AQI_FIPS['Unhealthy for Sensitive Groups Days'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Unhealthy Days Pct'] = AQI_FIPS['Unhealthy Days'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Very Unhealthy Days Pct'] = AQI_FIPS['Very Unhealthy Days'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Hazardous Days Pct'] = AQI_FIPS['Hazardous Days'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Days CO Pct'] = AQI_FIPS['Days CO'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Days NO2 Pct'] = AQI_FIPS['Days NO2'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Days Ozone Pct'] = AQI_FIPS['Days Ozone'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Days SO2 Pct'] = AQI_FIPS['Days SO2'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Days PM2.5 Pct'] = AQI_FIPS['Days PM2.5'] / AQI_FIPS['Days with AQI']
    AQI_FIPS['Days PM10 Pct'] = AQI_FIPS['Days PM10'] / AQI_FIPS['Days with AQI']
    return AQI_FIPS

def aqiPreprocessor(train_feature_years, predict_feature_year_lst, base_path):
    #AQI_paths = glob.glob(base_path + '/Data/AQI/*.zip')
    AQI_paths = [base_path+'/Data/AQI/annual_aqi_by_county_'+year+'.zip' 
                 for year in train_feature_years+predict_feature_year_lst]
    prev_8_AQI_paths = [base_path+'/Data/AQI/annual_aqi_by_county_'+str(int(year)-8)+'.zip' 
                        for year in train_feature_years+predict_feature_year_lst]
    prev_16_AQI_paths = [base_path+'/Data/AQI/annual_aqi_by_county_'+str(int(year)-16)+'.zip' 
                        for year in train_feature_years+predict_feature_year_lst]
    
    
    # Preprocess current year data
    AQI = readAqi(AQI_paths)
    county = readCountyFips(base_path)
    AQI_FIPS = mergyAqiFips(AQI, county)
    AQI_FIPS = dayToPercent(AQI_FIPS)
    AQI_FIPS.to_csv(base_path + '/Data/Data_with_FIPS/AQI_FIPS.csv')
    
    # Preprocess previous 8 years data
    prev_8_AQI = readAqi(prev_8_AQI_paths)
    county = readCountyFips(base_path)
    prev_8_AQI_FIPS = mergyAqiFips(prev_8_AQI, county)
    prev_8_AQI_FIPS = dayToPercent(prev_8_AQI_FIPS)
    prev_8_AQI_FIPS.rename(columns={'Year': 'prev_8_year'}, inplace=True)
    prev_8_AQI_FIPS['year'] = prev_8_AQI_FIPS['prev_8_year'] + 8
    prev_8_AQI_FIPS.to_csv(base_path + '/Data/Data_with_FIPS/prev_8_AQI_FIPS.csv')
    
    # Preprocess previous 16 years data
    prev_16_AQI = readAqi(prev_16_AQI_paths)
    county = readCountyFips(base_path)
    prev_16_AQI_FIPS = mergyAqiFips(prev_16_AQI, county)
    prev_16_AQI_FIPS = dayToPercent(prev_16_AQI_FIPS)
    prev_16_AQI_FIPS.rename(columns={'Year': 'prev_16_year'}, inplace=True)
    prev_16_AQI_FIPS['year'] = prev_16_AQI_FIPS['prev_16_year'] + 16
    prev_16_AQI_FIPS.to_csv(base_path + '/Data/Data_with_FIPS/prev_16_AQI_FIPS.csv')
    
    return AQI_FIPS, prev_8_AQI_FIPS, prev_16_AQI_FIPS
    