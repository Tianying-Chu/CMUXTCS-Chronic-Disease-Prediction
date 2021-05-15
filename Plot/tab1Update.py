# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 22:33:49 2021

@author: 13683
"""
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import os
import dash_table

from Model.RNN.modelSHAP import rnnForward
from Model.NN.model import nNet
import shap
import torch
import os
import joblib
import matplotlib.pyplot as plt
import base64
from sklearn import metrics

def get_cor(shap_values,df):
    cor_list=list()
    for i in range(shap_values.shape[1]):
        cor_list.append(np.corrcoef(shap_values[:,i], df.iloc[:,i])[0][1])
    return cor_list

def cumulativeImportance(model_df, shap_values):
    df=pd.DataFrame()
    df['feature']=model_df.columns
    df['importance']=abs(shap_values).sum(axis=0)
    df['cor']=get_cor(shap_values,model_df)
    df['Correlation']=(df['cor']>=0).replace(False,'Negative Correlated').replace(True,'Positive Correlated')
    df['abs'] = abs(df['importance'])
    df = df.sort_values(by='abs',ascending=False)[:10]
    df = df.sort_values(by='abs',ascending=True)
    return df

def change_feature_name(x,d):
    if x in d:
        return d[x]
    else:
        return x

def generate_table(dataframe, max_rows=15):
    df = dataframe.round(2)
    return     (dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= max_rows,
        style_data={'whiteSpace': 'normal','height': 'auto','minWidth': '180px', 'width': '180px', 'maxWidth': '180px',}
    ))

def importance_plot(model_type,model_name,base_path):
    shap.initjs()
    # For RNN
    if model_name == 'RNN':
        rnn = rnnForward().double()
        rnn.load_state_dict(torch.load(base_path+'/Result/'+model_name+'/'+model_type+'.shap'))
        rnn_df = pd.read_csv(base_path+'/Result/'+model_name+'/'+model_type+'.csv', index_col=0)

        feature_doc_df = pd.read_csv(base_path+'/Data/feature_documentation.csv')
        feature_dict = dict(zip(feature_doc_df['var_name'], feature_doc_df['short_name']))
        rnn_df.rename(columns=lambda x:change_feature_name(x,feature_dict),inplace=True)

        shap_values = shap.DeepExplainer(rnn, torch.tensor(rnn_df.values)).shap_values(torch.tensor(rnn_df.values))
        summaryplot = shap.summary_plot(shap_values, rnn_df,show=False)
        plt.savefig('temp.png',bbox_inches = 'tight')
        plt.close()
        
        df = cumulativeImportance(rnn_df, shap_values)
        
    # For NN
    elif model_name == 'NN':
        n_nn= 1000#number of lines want to look
        nn = nNet().double()
        nn.load_state_dict(torch.load(base_path+'/Model/'+model_name+'/'+model_type))
        nn_df = pd.read_csv(base_path+'/Data/Transformed/'+model_type+'_transformed_800.csv', index_col=0)
        cols = [col for col in nn_df.columns if col not in ['fips', 'value', 'year']]
        nn_df = nn_df[cols]
        
        feature_doc_df = pd.read_csv(base_path+'/Data/feature_documentation.csv')
        feature_dict = dict(zip(feature_doc_df['var_name'], feature_doc_df['short_name']))
        nn_df.rename(columns=lambda x:change_feature_name(x,feature_dict),inplace=True)
        
        shap_values = shap.DeepExplainer(nn, torch.tensor(nn_df.values[:n_nn])).shap_values(torch.tensor(nn_df.values[:n_nn]))
        summaryplot = shap.summary_plot(shap_values, nn_df[:n_nn],show=False)
        plt.savefig('temp.png',bbox_inches = 'tight')
        plt.close()
        
        df = cumulativeImportance(nn_df[:n_nn], shap_values)
        
    # For Random Forest
    elif model_name =='RF':
        n_rf =1000
        rf_df = pd.read_csv(base_path+'/Data/Transformed/'+model_type+'_transformed_800.csv', index_col=0)
        cols = [col for col in rf_df.columns if col not in ['fips', 'value', 'year']]
        rf_df = rf_df[cols]
        
        feature_doc_df = pd.read_csv(base_path+'/Data/feature_documentation.csv')
        feature_dict = dict(zip(feature_doc_df['var_name'], feature_doc_df['short_name']))
        rf_df.rename(columns=lambda x:change_feature_name(x,feature_dict),inplace=True)
        
        rf = joblib.load(base_path+'/Model/'+model_name+'/'+model_type)
        shap_values = shap.TreeExplainer(rf,rf_df.values[:n_rf]).shap_values(rf_df.values[:n_rf],check_additivity=False)
        summaryplot = shap.summary_plot(shap_values, rf_df[:n_rf],show=False)
        plt.savefig('temp.png',bbox_inches = 'tight')
        plt.close()
        
        df = cumulativeImportance(rf_df[:n_rf], shap_values)
                
    # For KNN
    # Must use Kernel method on knn
    elif model_name == 'KNN':
        n_knn=10
        knn_df = pd.read_csv(base_path+'/Data/Transformed/'+model_type+'_transformed_800.csv', index_col=0)
        cols = [col for col in knn_df.columns if col not in ['fips', 'value', 'year']]
        knn_df = knn_df[cols]
        
        feature_doc_df = pd.read_csv(base_path+'/Data/feature_documentation.csv')
        feature_dict = dict(zip(feature_doc_df['var_name'], feature_doc_df['short_name']))
        knn_df.rename(columns=lambda x:change_feature_name(x,feature_dict),inplace=True)
        
        knn = joblib.load(base_path+'/Model/'+model_name+'/'+model_type)
        shap_values = shap.KernelExplainer(knn.predict, knn_df.values[:n_knn]).shap_values(knn_df.values[:n_knn])
        summaryplot = shap.summary_plot(shap_values, knn_df[:n_knn],show=False)
        plt.savefig('temp.png',bbox_inches = 'tight')
        plt.close()
        
        df = cumulativeImportance(knn_df[:n_knn], shap_values)
                
    # For LR
    elif model_name == 'LR':
        n_lr=1000
        lr_df = pd.read_csv(base_path+'/Data/Transformed/'+model_type+'_transformed_800.csv', index_col=0)
        cols = [col for col in lr_df.columns if col not in ['fips', 'value', 'year']]
        lr_df = lr_df[cols]
        
        feature_doc_df = pd.read_csv(base_path+'/Data/feature_documentation.csv')
        feature_dict = dict(zip(feature_doc_df['var_name'], feature_doc_df['short_name']))
        lr_df.rename(columns=lambda x:change_feature_name(x,feature_dict),inplace=True)
        
        lr = joblib.load(base_path+'/Model/'+model_name+'/'+model_type)
        shap_values = shap.LinearExplainer(lr, lr_df.values[:n_lr]).shap_values(lr_df.values[:n_lr])
        summaryplot = shap.summary_plot(shap_values, lr_df[:n_lr],show=False)
        plt.savefig('temp.png',bbox_inches = 'tight')
        plt.close()
        
        df = cumulativeImportance(lr_df[:n_lr], shap_values)
        
    encoded_image = base64.b64encode(open('temp.png', 'rb').read()).decode('ascii')
    return [html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image),style={'width': '80%', 'height':'80%'})])], px.bar(df,x='importance',y='feature',color='Correlation',category_orders={'feature':list(df['feature'].iloc[::-1])})



def tab1Update(base_path, df, eval_df, predict_label_year, model_type, model_name):
    cohort_year = str(int(predict_label_year[:4])-1)
    df_predict_year = df[df['year']==int(cohort_year)]
    df_predict_year['fips'] =df['fips'].apply(lambda x: str(x).zfill(5))
    df_predict_year['fips'].astype(str)
    if not os.path.exists(os.path.join(base_path, 'Result/Prediction')):
        os.makedirs(os.path.join(base_path, 'Result/Prediction'))
    df_predict_year.to_csv(base_path + '/Result/Prediction/Prediction_{}_{}_{}.csv'.format(model_type, model_name, predict_label_year))

    #table
    table_df = df_predict_year[['fips','state','county','value_hat']]
    table_df['value_hat'] = df['value_hat'].apply(lambda x: round(x, 2))
    table_df.rename(columns={"value_hat": "predicted hospitalization rate"},inplace=True)
    df_table =generate_table(table_df)
    
    #prediction map
    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    map_us = px.choropleth(df_predict_year, 
                        geojson=counties,
                        locations='fips',
                        color='value_hat',
                        color_continuous_scale="viridis",
                        #range_color=(20, 60),
                        scope="usa",
                        hover_data=['state','county'],
                        labels={'value_hat':'Predicted value'})
    map_us.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
    #model metrics
    latest_year = eval_df.year.max()
    eval_df = eval_df[eval_df['year']==latest_year]
    eval_df['abs_diff']=abs(eval_df['value_hat']-eval_df['value'])
    actual_predict_plot = px.scatter(eval_df,x='value_hat', y='value',color='abs_diff',
                                     labels=dict(value_hat="Predicted hospitalization rate", 
                                                 value="Actual hospitalization rate",
                                                 abs_diff='|actual - predicted|'),
                                     hover_data=['state','county'],)
    actual_predict_plot.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 
                                                 'line_color': 'red',
                                                 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}])
    actual_predict_plot.update_layout(xaxis_range=[eval_df['value'].min()-5,eval_df['value'].max()+5],
                                      yaxis_range=[eval_df['value'].min()-5,eval_df['value'].max()+5])
    
    r2 = 'Unadjusted R square for the model is: '+str(round(metrics.r2_score(eval_df['value'], eval_df['value_hat']),2))
    model_metrics = [html.Div(id='R2metric', children=r2),
                     dcc.Graph(id='actual-predict-plot', figure=actual_predict_plot)]
    
    feature_contribution, feature_importance = importance_plot(model_type,model_name,base_path)
    
    return df_table,map_us,model_metrics, feature_contribution, feature_importance