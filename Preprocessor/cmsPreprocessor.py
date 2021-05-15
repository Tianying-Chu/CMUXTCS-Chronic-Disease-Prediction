# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 08:52:27 2021

@author: Tianying Chu
"""

import pandas as pd
import zipfile

def addYear(df,year):
    # Add year column
    df= df.drop(df.index[0])
    df['Year'] = year
    cols = df.columns.tolist()
    cols.insert(3, cols.pop(cols.index('Year')))
    df = df.reindex(columns= cols)
    return df

def mergeYears(zip_file, table):
    paths = zip_file.namelist()
    for i in range(len(paths)):
        year = paths[i][-9:-5]
        if i == 0:
            #merged = pd.read_csv(paths[i])
            xlsx_file = zip_file.open(paths[i])
            merged = pd.read_excel(xlsx_file, table, header=4)
            merged = addYear(merged, year)
        else:
            xlsx_file = zip_file.open(paths[i])
            df = pd.read_excel(xlsx_file, table, header=4)
            df = addYear(df, year)
            merged = pd.concat([merged, df], axis=0)
    merged.reset_index(drop=True, inplace=True)
    return merged

def transformCms(df):
    df.rename(columns={'Unnamed: 0': 'State', 'Unnamed: 1': 'County','Unnamed: 2':'FIPS','Unnamed: 4':"Alzheimer's Disease/Dementia "}, inplace=True)
    df= df[df.County != '  ']
    df = df.dropna(subset=['County'])
    df= df[df.County != 'Unknown ']
    df = df.reset_index(drop=True)
    df=df.rename(columns={'Unnamed: 17':'Hepatitis (Chronic Viral B & C)',
                          'Hepatitis                                 (Chronic Viral B & C)':'Hepatitis (Chronic Viral B & C)'})
    return df

def readCms(CMS_paths, table):
    zip_file = zipfile.ZipFile(CMS_paths, 'r')
    #print(zip_file.namelist())
    CMS = mergeYears(zip_file, table)
    CMS = transformCms(CMS)
    CMS.drop_duplicates(['FIPS', 'Year'], keep='first', inplace=True)
    return CMS

def cmsPreprocessor(base_path):
    prevalence_paths = base_path + '/Data/CMS/County_Table_Chronic_Conditions_Prevalence_by_Age.zip'
    spending_paths = base_path + '/Data/CMS/County_Table_Chronic_Conditions_Spending.zip'
    
    prevalence = readCms(prevalence_paths, 'Beneficiaries 65 Years and Over')
    spending = readCms(spending_paths, 'Actual Spending')
    
    prevalence.to_csv(base_path + '/Data/Data_with_FIPS/Prevalence_FIPS.csv')
    spending.to_csv(base_path + '/Data/Data_with_FIPS/Spending_FIPS.csv')
    return