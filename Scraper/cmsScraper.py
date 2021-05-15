# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:35:58 2021

@author: Tianying Chu
"""
import os
from selenium import webdriver
import time
import zipfile

def checkCMSYears(CMS_paths):
    zip_file = zipfile.ZipFile(CMS_paths, 'r')
    paths = zip_file.namelist()
    paths.sort()
    latest_CMS = paths[-1][-9:-5]
    return int(latest_CMS)

def getLatestYear(prevalence_paths, spending_paths):
    latest_prevalence = checkCMSYears(prevalence_paths)
    latest_spending = checkCMSYears(spending_paths)
    latest_CMS = min(latest_prevalence, latest_spending)
    return latest_CMS

def cmsScraper(train_feature_years, predict_feature_year, base_path):
    # Set download directory
    chromeOptions = webdriver.ChromeOptions()
    prefs = {'download.default_directory': base_path.replace('/', '\\') + '\\Data\\CMS', 
             'profile.default_content_setting_values.automatic_downloads': 1}
    chromeOptions.add_experimental_option('prefs', prefs)
    chromeOptions.add_argument('headless')
    chromeOptions.add_argument('--no-sandbox')
    
    # Connect to CMS website
    driver = webdriver.Chrome(executable_path=base_path+'\chromedriver.exe', 
                              chrome_options=chromeOptions)
    driver.get("https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Chronic-Conditions/CC_Main")
    
    # Download CMS files
    if not os.path.exists(os.path.join(base_path, 'Data/CMS/County_Table_Chronic_Conditions_Spending.zip')):
        driver.find_element_by_link_text('Spending County Level: All Beneficiaries, 2007-2018 (ZIP)').click()
        time.sleep(5)
    if not os.path.exists(os.path.join(base_path, 'Data/CMS/County_Table_Chronic_Conditions_Prevalence_by_Age.zip')):
        driver.find_element_by_link_text('Prevalence State/County Level: All Beneficiaries by Age, 2007-2018 (ZIP)').click()
        time.sleep(5)
    driver.quit()
    
    # Get latest year
    prevalence_paths = base_path + '/Data/CMS/County_Table_Chronic_Conditions_Prevalence_by_Age.zip'
    spending_paths = base_path + '/Data/CMS/County_Table_Chronic_Conditions_Spending.zip'
    latest_CMS = getLatestYear(prevalence_paths, spending_paths)
    
    return latest_CMS