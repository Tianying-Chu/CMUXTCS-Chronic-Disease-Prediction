# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:50:21 2020

@author: 13683
"""

#pip install selenium
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import time
from selenium.webdriver.common.by import By
import pandas as pd
import re
import os

def get_df(driver, year):
    row_path = "//div[@id='report_type_tabs']/div[@id='tabReport']/div[@id='tabReport_content']/div[@id='div_report_result']/div[@id='report_tabs']/div[@id='tab2']/div[@id='tab2_content']/table[@id='report_table']/tbody[@id='report_body']/tr"
    rows = driver.find_elements(By.XPATH,row_path)
    l=list()
    for row in rows:
        try:        
            s= row.get_attribute("innerHTML")
            slist = s.split('<td>')
            s1=re.search('>[a-zA-Z\s]+<',slist[0])[0].strip('>').strip('<')
            s2=slist[1].strip('</td>')
            s3=slist[2].strip('</td>')
            s4=slist[4].strip('</td>')
            #print(s1,s2,s3,s4)
        except:
            pass
        l.append([s1,s2,s3,s4,year])

    row_path = "//div[@id='report_type_tabs']/div[@id='tabReport']/div[@id='tabReport_content']/div[@id='div_report_result']/div[@id='report_tabs']/div[@id='tab2']/div[@id='tab2_content']/table[@id='insufficientData_table']/tbody[@id='insufficientData_body']//tr"
    rows = driver.find_elements(By.XPATH,row_path)
    for row in rows:
        try:        
            s= row.get_attribute("innerHTML")
            slist = s.split('<td>')
            s1=re.search('>[a-zA-Z\s]+<',slist[0])[0].strip('>').strip('<')
            s2=slist[1].strip('</td>')
            s3=slist[2].strip('</td>')
            s4=slist[4].strip('</td>')
            #print(s1,s2,s3,s4)
        except:
            pass
        l.append([s1,s2,s3,s4,year])
    df = pd.DataFrame(l)
    df.columns = ['County','state','value','range','year']
    return df

def get_latest_year(driver, cat):
    year_list =list()
    category_drop = Select(driver.find_element_by_id("select_health_indicator"))
    time.sleep(2)
    category_drop.select_by_visible_text(cat)
    health_drop = Select(driver.find_element_by_id("select_indicator_theme"))
    time.sleep(2)
    health_drop.select_by_visible_text('Hospitalizations')
    year_drop = Select(driver.find_element_by_id("select_1"))
    time.sleep(2)
    for option in year_drop.options:
        year_list.append(option.text)
    return max(year_list)

def hospitalizationScraper(model_type, train_label_years, base_path):
    year_options = []
    for year in train_label_years:
        if not os.path.exists(os.path.join(base_path, 'Data/Heart_Disease/All Heart Disease{}.csv'.format(year))):
            year_options.append(year)
            
    option = webdriver.ChromeOptions()
    option.add_argument("headless")
    option.add_argument('--no-sandbox')
    driver = webdriver.Chrome(executable_path = base_path+"\chromedriver.exe", options=option)
    driver.get("https://nccd.cdc.gov/DHDSPAtlas/Reports.aspx")
    
    if model_type == 'heart':
        latest_label_year = get_latest_year(driver, 'All Heart Disease')
    elif model_type == 'stroke':
        latest_label_year = get_latest_year(driver, 'All Stroke')
    
    category_options = ['All Heart Disease','All Stroke']
    health_options = ['Hospitalizations']
    #year_options = ['2008-2010','2009-2011','2010-2012','2011-2013','2012-2014','2013-2015','2014-2016','2015-2017']
    race_options = ['All Races/Ethnicities']
    gender_options = ['Both Genders']
    age_options = ['65+']
    smooth_options = ['Not Smoothed']
    
    for cat in category_options:
        for heal in health_options:
            for year in year_options:
                for race in race_options:
                    for gender in gender_options:
                        for age in age_options:
                            for smooth in smooth_options:
                                print(cat,heal,year,race,gender,age,smooth)
                                category_drop = Select(driver.find_element_by_id("select_health_indicator"))
                                time.sleep(2)
                                category_drop.select_by_visible_text(cat)
                                time.sleep(2)
                                health_drop = Select(driver.find_element_by_id("select_indicator_theme"))
                                health_drop.select_by_visible_text(heal)
                                time.sleep(2)
                                year_drop = Select(driver.find_element_by_id("select_1"))
                                year_drop.select_by_visible_text(year)
                                time.sleep(2)
                                race_drop = Select(driver.find_element_by_id("select_5"))
                                race_drop.select_by_visible_text(race)
                                time.sleep(2)
                                gender_drop = Select(driver.find_element_by_id("select_3"))
                                gender_drop.select_by_visible_text(gender)
                                time.sleep(2)
                                age_drop = Select(driver.find_element_by_id("select_6"))
                                age_drop.select_by_visible_text(age)
                                time.sleep(2)
                                smooth_drop = Select(driver.find_element_by_id("select_7"))
                                smooth_drop.select_by_visible_text(smooth)
                                buttonSearch = driver.find_element_by_id("btn_report_submit")
                                buttonSearch.click()
                                time.sleep(10)
                                df=get_df(driver, year)
                                if cat == 'All Heart Disease':
                                    df.to_csv(base_path + '/Data/Heart_Disease/'+cat+year+'.csv',index=False)
                                elif cat == 'All Stroke':
                                    df.to_csv(base_path + '/Data/Stroke/'+cat+year+'.csv',index=False)
    time.sleep(2)
    driver.quit()
    return latest_label_year