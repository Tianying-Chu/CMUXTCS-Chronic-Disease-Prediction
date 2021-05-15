# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:42:38 2021

@author: Tianying Chu
"""
import os
from selenium import webdriver
import time

def aqiScraper(train_feature_years, predict_feature_year_lst, base_path):
    # Set download directory
    chromeOptions = webdriver.ChromeOptions()
    prefs = {'download.default_directory': base_path.replace('/', '\\') + '\\Data\\AQI', 
             'profile.default_content_setting_values.automatic_downloads': 1}
    chromeOptions.add_experimental_option('prefs', prefs)
    chromeOptions.add_argument('headless')
    chromeOptions.add_argument('--no-sandbox')
    
    # Connect to AQI website
    driver = webdriver.Chrome(executable_path=base_path+'\chromedriver.exe', 
                              chrome_options=chromeOptions)
    driver.get("https://aqs.epa.gov/aqsweb/airdata/download_files.html#Annual")
    
    train_prev_8_years = [str(int(year)-8) for year in train_feature_years]
    train_prev_16_years = [str(int(year)-16) for year in train_feature_years]
    predict_prev_8_years = [str(int(year)-8) for year in predict_feature_year_lst]
    predict_prev_16_years = [str(int(year)-16) for year in predict_feature_year_lst]
    
    all_years = list(set(train_feature_years + train_prev_8_years + train_prev_16_years +
                         predict_feature_year_lst + predict_prev_8_years + predict_prev_16_years))
    all_years.sort()
    
    # Download AQI files
    latest_AQI = None
    for year in all_years:
        try:
            if not os.path.exists(os.path.join(base_path, 'Data/AQI/annual_aqi_by_county_{}.zip'.format(year))):
                driver.find_element_by_link_text('annual_aqi_by_county_'+year+'.zip').click()
                time.sleep(2)
            latest_AQI = year
        except:
            break
    driver.quit()
    return int(latest_AQI)

