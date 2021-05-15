# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:59:27 2021

@author: Tianying Chu
"""

from Scraper.hospitalizationScraper import hospitalizationScraper
from Scraper.acsScraper import acsScraper
from Scraper.aqiScraper import aqiScraper
from Scraper.cmsScraper import cmsScraper
from Preprocessor.hospitalizationPreprocessor import hospitalizationPreprocessor
from Preprocessor.acsPreprocessor import acsPreprocessor
from Preprocessor.aqiPreprocessor import aqiPreprocessor
from Preprocessor.cmsPreprocessor import cmsPreprocessor
from combiner import combiner
from imputer import imputer
from transformer import transformer


def dataUpdate(base_path, model_type, predict_label_year, upload_cms):
    # Scrape data from different sources and get latest year
    predict_feature_year = str(int(predict_label_year[:4]) - 1)
    train_label_years = [str(i+1) + '-' + str(i+3)
                        for i in range(2010, int(predict_feature_year)-3, 1)]
    latest_label_year = hospitalizationScraper(model_type, train_label_years, base_path)
    predict_feature_year_lst = [str(i) for i in range(int(latest_label_year[:4]), int(predict_label_year[:4]))] # Yilun
    train_feature_years = [str(i) for i in range(2010, int(predict_feature_year)-3, 1)]
    
    latest_ACS = acsScraper(train_feature_years, predict_feature_year_lst, base_path)
    latest_AQI = aqiScraper(train_feature_years, predict_feature_year_lst, base_path)
    if upload_cms is None:
        latest_CMS = cmsScraper(train_feature_years, predict_feature_year, base_path)
    else: 
        latest_CMS = int(predict_feature_year)
    
    # Compare latest_year with predict_feature_year
    latest_year = min(latest_ACS, latest_AQI, latest_CMS)
    if latest_year < int(predict_feature_year):
        latest_year_message = 'Unable to fetch the latest year!\n'
        if latest_ACS < int(predict_feature_year):
            latest_year_message += 'ACS data is only available till {}!\n'.format(latest_ACS)
        if latest_AQI < int(predict_feature_year):
            latest_year_message += 'AQI data is only available till {}!\n'.format(latest_AQI)
        if (latest_CMS < int(predict_feature_year)) & (upload_cms is None):
            latest_year_message += 'CMS data is only available till {}!\n'.format(latest_CMS)
        
    else:
        # Data preprocessing for each data sources
        hospitalizationPreprocessor(train_label_years, base_path)
        acsPreprocessor(base_path)
        aqiPreprocessor(train_feature_years, predict_feature_year_lst, base_path)
        if upload_cms is None:
            cmsPreprocessor(base_path)
        
        # Merge data sources
        combiner(train_feature_years, predict_feature_year_lst, base_path)
        
        # Impute missing values and transform columns
        imputer(train_feature_years, predict_feature_year, base_path)
        transformer(train_feature_years, predict_feature_year, base_path)
        latest_year_message = 'Successfully update data to the latest year!'
    
    return latest_year_message, train_feature_years, predict_feature_year