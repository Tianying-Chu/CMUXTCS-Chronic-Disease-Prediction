# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:04:48 2021

@author: sneti
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#county is county input, state is state input, df is from the 
def countyHistoricalTrends(state_value, county_value, train_df, predict_df):
    state_train = train_df[train_df['state'] == state_value]
    county_train = state_train[state_train['county'] == county_value]
    state_predict = predict_df[predict_df['state'] == state_value]
    county_predict = state_predict[state_predict['county'] == county_value]
    
    # National average
    nation_train_yravg = train_df.groupby('year', as_index=False).agg({'value':'mean', 'value_hat':'mean'})
    nation_predict_yravg = predict_df.groupby('year', as_index=False).agg({'value_hat':'mean'})
    label_years = county_train['year'].append(county_predict['year']).apply(lambda x: str(int(x)+1)+'-'+str(int(x)+3))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=label_years, 
                             y=county_train['value'],
                             mode='lines+markers',
                             name='Actual Values')) 
    fig.add_trace(go.Scatter(x=label_years, 
                             y=county_train['value_hat'].append(county_predict['value_hat']),
                             line = dict(dash='dash'),
                             name='Predicted Values'))
    fig.add_trace(go.Scatter(x=label_years, 
                             y=nation_train_yravg['value_hat'].append(nation_predict_yravg['value_hat']),
                             line = dict(dash='dash'),
                             name='Predicted National Average'))
    
    fig.update_layout(title='Historical Trends for Hospitalization Rate for ' + county_value + ", " + state_value)
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='Hospitalization Rate (per 1000 Medicare beneficiaries)')
    fig.update_layout(showlegend=True)
    return fig

def countyFeaturePlot(state_value, county_value, feature, train_df, predict_df, base_path):
    feature_df = pd.read_csv(base_path+'/Data/feature_documentation.csv')
    feat_name = feature_df[feature_df['var_name'] == feature]['short_name'].tolist()[0]
    datasrc = feature_df[feature_df['var_name'] == feature]['data_source'].tolist()[0]
    aggby = feature_df[feature_df['var_name'] == feature]['agg_methods'].tolist()[0]
    units = feature_df[feature_df['var_name'] == feature]['units'].tolist()[0]
    
    if aggby == 'weighted mean(over 65)':
        train_df['weightedavg_value'] = train_df[feature] * train_df['population_65_county']
        natavg_train = train_df.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_65_county':'sum'})
        natavg_train['feature_weighted'] = natavg_train['weightedavg_value'] / natavg_train['population_65_county']
        
        predict_df['weightedavg_value'] = predict_df[feature] * predict_df['population_65_county']
        natavg_predict = predict_df.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_65_county':'sum'})
        natavg_predict['feature_weighted'] = natavg_predict['weightedavg_value'] / natavg_predict['population_65_county']
        natyaxis = natavg_train['feature_weighted'].append(natavg_predict['feature_weighted'])
    
    elif aggby == 'weighted mean(all)':
        train_df['weightedavg_value'] = train_df[feature] * train_df['population_all_county']
        natavg_train = train_df.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_all_county':'sum'})
        natavg_train['feature_weighted'] = natavg_train['weightedavg_value'] / natavg_train['population_all_county']
        
        predict_df['weightedavg_value'] = predict_df[feature] * predict_df['population_all_county']
        natavg_predict = predict_df.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_all_county':'sum'})
        natavg_predict['feature_weighted'] = natavg_predict['weightedavg_value'] / natavg_predict['population_all_county']
        natyaxis = natavg_train['feature_weighted'].append(natavg_predict['feature_weighted'])
    
    elif aggby == 'mean':
        natavg_train = train_df.groupby('year', as_index=False).agg({feature:'mean'})
        natavg_predict = predict_df.groupby('year', as_index=False).agg({feature:'mean'})
        natyaxis = natavg_train[feature].append(natavg_predict[feature])
    
    '''
    natavg_train = train_df.groupby('year', as_index=False).agg({feature:'mean'})
    natavg_predict = predict_df.groupby('year', as_index=False).agg({feature:'mean'})
    '''
    state_train = train_df[train_df['state'] == state_value]
    county_train = state_train[state_train['county'] == county_value]
    state_predict = predict_df[predict_df['state'] == state_value]
    county_predict = state_predict[state_predict['county'] == county_value]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=county_train['year'].append(county_predict['year']), 
                             y=county_train[feature].append(county_predict[feature]),
                             mode='lines+markers',
                             name=state_value+', '+county_value))
    fig.add_trace(go.Scatter(x=natavg_train['year'].append(natavg_predict['year']),
                             y = natyaxis,
                             line = dict(dash='dash'),
                             name = 'National Average'))
    
    #fig.update_layout(title='Historical Trends for ' + feat_name + ' for ' + county_value + ", " + state_value)
    fig.update_xaxes(title='Year')
    fig.update_layout(title=feat_name.title() + ' (in '+ units + ')')
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(yanchor="bottom", y=0.01,
                                  xanchor="left", x=0.01))
    return fig

    