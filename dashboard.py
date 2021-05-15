# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 22:37:13 2021

@author: Tianying Chu
"""
import io
import base64
from zipfile import ZipFile
from Preprocessor.cmsPreprocessor import readCms

import pandas as pd

# Process uploaded file
def decode_zips(base_path, contents, filename):
    content_type, content_string = contents.split(',')
    content_decoded = base64.b64decode(content_string)
    zip_str = io.BytesIO(content_decoded)
    if 'spending' in filename.lower():
        CMS = readCms(zip_str, 'Actual Spending')
        CMS.to_csv(base_path + '/Data/Data_with_FIPS/Spending_FIPS.csv')
    if 'prevalence' in filename.lower():
        CMS = readCms(zip_str, 'Beneficiaries 65 Years and Over')
        CMS.to_csv(base_path + '/Data/Data_with_FIPS/Prevalence_FIPS.csv')
    return None

def rename_feature(base_path, train_df, predict_df):
    return