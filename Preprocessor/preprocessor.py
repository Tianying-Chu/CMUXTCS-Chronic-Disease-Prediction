# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:23:11 2021

@author: Tianying Chu
"""
import pandas as pd
import re

def specialCase(string):
    if not re.match('roanoke|baltimore', string.lower()):
        return re.sub(' (County|City|Borough|Census Area|city|City County|Municipality|City and Borough|Parish|\(City\))$', '', string)
    else:
        return re.sub(' (County|Borough|Census Area|Municipality|City and Borough|Parish|\(City\))$', '', string)
        
def countyNamePreprocessor(df):
    #df['County'] = df['County'].str.replace(' (County|City|Borough|Census Area|city|City County|Municipality|City and Borough|Parish|\(City\))$', '')
    df['County'] = df['County'].apply(specialCase)
    df['County'] = df['County'].str.strip()
    df['County'] = df['County'].str.title()
    df['County'] = df['County'].str.replace('Saint[e]?|St[e]?\.', 'St')
    df['County'] = df['County'].str.replace('Wrangell Petersburg', 'Petersburg')
    df['County'] = df['County'].str.replace('La Salle', 'Lasalle')
    return df

def readCountyFips(base_path):
    # Read county data
    county_fips_path = base_path + '/Data/my_county_fips_master.csv'
    county = pd.read_csv(county_fips_path, encoding='unicode_escape').iloc[:, :5]
    county.rename(columns={'_name': 'County'}, inplace=True)
    county = countyNamePreprocessor(county)
    county['fips'] = county['geo_id'].str[-5:]
    return county

def mergeYears(paths):
    for i in range(len(paths)):
        if i == 0:
            merged = pd.read_csv(paths[i])
        else:
            df = pd.read_csv(paths[i])
            merged = pd.concat([merged, df], axis=0)
    merged.reset_index(drop=True, inplace=True)
    return merged
