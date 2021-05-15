# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:27:23 2021

@author: Tianying Chu
"""

from cohort import cohort
from train import train
from testMetrics import testMetrics
from predict import prediction

def pipeline(base_path, model_type, predict_label_year, train_feature_years, predict_feature_year, load_model, model_name, upload_cms):
    
    # Get interested cohorts
    if load_model == 'Yes':
        load_model = True
    else:
        load_model = False
    cohorts = cohort(model_name, train_feature_years, predict_feature_year, base_path)
    
    # Train the baseline model
    model = train(model_name, load_model, cohorts, model_type, base_path)
    
    # Get the metrics
    testMetrics(model, model_name, model_type, cohorts, base_path)
    
    # Make prediction
    train_df, predict_df = prediction(model, model_name, model_type, base_path)
    
    return model, train_df, predict_df
