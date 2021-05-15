# -*- coding: utf-8 -*-
"""
Created on Tue May  4 00:29:37 2021

@author: sneti
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#county is county input, state is state input, df is from the 
def stateHistoricalTrends(state_value, train_df, predict_df):
    state_train = train_df[train_df['state'] == state_value]
    state_train_yravg = state_train.groupby('year', as_index=False).agg({'value':'mean', 'value_hat':'mean'})
    state_predict = predict_df[predict_df['state'] == state_value]
    state_predict_yravg = state_predict.groupby('year', as_index=False).agg({'value_hat':'mean'})
    
    # National average
    nation_train_yravg_county = train_df.groupby(['state', 'year'], as_index=False).agg({'value':'mean', 'value_hat':'mean'})
    nation_train_yravg = nation_train_yravg_county.groupby('year', as_index=False).agg({'value':'mean', 'value_hat':'mean'})
    nation_predict_yravg_county = predict_df.groupby(['state', 'year'], as_index=False).agg({'value_hat':'mean'})
    nation_predict_yravg = nation_predict_yravg_county.groupby('year', as_index=False).agg({'value_hat':'mean'})
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=state_train_yravg['year'].append(state_predict_yravg['year']).apply(lambda x: str(int(x)+1)+'-'+str(int(x)+3)),
                             y=state_train_yravg['value'],
                             mode='lines+markers',
                             name='Actual Values'))
    fig.add_trace(go.Scatter(x=state_train_yravg['year'].append(state_predict_yravg['year']).apply(lambda x: str(int(x)+1)+'-'+str(int(x)+3)), 
                             y=state_train_yravg['value_hat'].append(state_predict_yravg['value_hat']),
                             line = dict(dash='dash'),
                             name='Predicted Values'))
    fig.add_trace(go.Scatter(x=nation_train_yravg['year'].append(nation_predict_yravg['year']).apply(lambda x: str(int(x)+1)+'-'+str(int(x)+3)), 
                             y=nation_train_yravg['value_hat'].append(nation_predict_yravg['value_hat']),
                             line = dict(dash='dash'),
                             name='Predicted National Average'))
    
    fig.update_layout(title='Historical Trends for Hospitalization Rate (averaged over all counties)')
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='Hospitalization Rate (per 1000 Medicare beneficiaries)')
    fig.update_layout(showlegend=True)
    return fig

def stateFeaturePlot(state_value, feature, train_df, predict_df, base_path):
    #pulling info from features df
    feature_df = pd.read_csv(base_path+'/Data/feature_documentation.csv')
    feat_name = feature_df[feature_df['var_name'] == feature]['short_name'].tolist()[0]
    datasrc = feature_df[feature_df['var_name'] == feature]['data_source'].tolist()[0]
    aggby = feature_df[feature_df['var_name'] == feature]['agg_methods'].tolist()[0]
    units = feature_df[feature_df['var_name'] == feature]['units'].tolist()[0]
    
    #creating national average
    natavg_train = train_df.groupby('year', as_index=False).agg({feature:'mean'})
    natavg_predict = predict_df.groupby('year', as_index=False).agg({feature:'mean'})
    
    #if agg method is sum, just add everything
    if aggby == 'sum':
        state_train = train_df[train_df['state'] == state_value]
        state_train_yravg = state_train.groupby('year', as_index=False).agg({feature:'sum'})
        state_predict = predict_df[predict_df['state'] == state_value]
        state_predict_yravg = state_predict.groupby('year', as_index=False).agg({feature:'sum'})
        
        title = 'Historical Trends for ' + feat_name + ' (summed over all counties)'
        yaxis = state_train_yravg[feature].append(state_predict_yravg[feature])
        
    elif aggby == 'weighted mean(over 65)':

        #multiply feature value with population
        state_train = train_df[train_df['state'] == state_value]
        state_train['weightedavg_value'] = state_train[feature] * state_train['population_65_county']

        #group by year by summing, then divide by summed population to get weighted average per year 
        state_train_yravg = state_train.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_65_county':'sum'})
        state_train_yravg['feature_weighted'] = state_train_yravg['weightedavg_value'] / state_train_yravg['population_65_county']

        #do it all over again for predict_df
        state_predict = predict_df[predict_df['state'] == state_value]
        state_predict['weightedavg_value'] = state_predict[feature] * state_predict['population_65_county']

        state_predict_yravg = state_predict.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_65_county':'sum'})
        state_predict_yravg['feature_weighted'] = state_predict_yravg['weightedavg_value'] / state_predict_yravg['population_65_county']

        title = 'Historical Trends for ' + feat_name + ' (population (over 65) weighted average over all counties)'
        yaxis = state_train_yravg['feature_weighted'].append(state_predict_yravg['feature_weighted'])
        
        # National average
        train_df['weightedavg_value'] = train_df[feature] * train_df['population_65_county']
        natavg_train = train_df.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_65_county':'sum'})
        natavg_train['feature_weighted'] = natavg_train['weightedavg_value'] / natavg_train['population_65_county']
        
        predict_df['weightedavg_value'] = predict_df[feature] * predict_df['population_65_county']
        natavg_predict = predict_df.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_65_county':'sum'})
        natavg_predict['feature_weighted'] = natavg_predict['weightedavg_value'] / natavg_predict['population_65_county']
        natyaxis = natavg_train['feature_weighted'].append(natavg_predict['feature_weighted'])
    
    elif aggby == 'weighted mean(all)':
        
        #multiply feature value with population
        state_train = train_df[train_df['state'] == state_value]
        state_train['weightedavg_value'] = state_train[feature] * state_train['population_all_county']
        
        #group by year by summing, then divide by summed population to get weighted average per year 
        state_train_yravg = state_train.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_all_county':'sum'})
        state_train_yravg['feature_weighted'] = state_train_yravg['weightedavg_value'] / state_train_yravg['population_all_county']
        
        #do it all over again for predict_df
        state_predict = predict_df[predict_df['state'] == state_value]
        state_predict['weightedavg_value'] = state_predict[feature] * state_predict['population_all_county']
        
        state_predict_yravg = state_predict.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_all_county':'sum'})
        state_predict_yravg['feature_weighted'] = state_predict_yravg['weightedavg_value'] / state_predict_yravg['population_all_county']
        
        title = 'Historical Trends for ' + feat_name + ' (population weighted average over all counties)'
        yaxis = state_train_yravg['feature_weighted'].append(state_predict_yravg['feature_weighted'])
        
        # National average
        train_df['weightedavg_value'] = train_df[feature] * train_df['population_all_county']
        natavg_train = train_df.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_all_county':'sum'})
        natavg_train['feature_weighted'] = natavg_train['weightedavg_value'] / natavg_train['population_all_county']
        
        predict_df['weightedavg_value'] = predict_df[feature] * predict_df['population_all_county']
        natavg_predict = predict_df.groupby('year', as_index=False).agg({'weightedavg_value':'sum', 'population_all_county':'sum'})
        natavg_predict['feature_weighted'] = natavg_predict['weightedavg_value'] / natavg_predict['population_all_county']                                    
        natyaxis = natavg_train['feature_weighted'].append(natavg_predict['feature_weighted'])

        
    elif aggby == 'mean':
        state_train = train_df[train_df['state'] == state_value]
        state_train_yravg = state_train.groupby('year', as_index=False).agg({feature:'mean'})
        state_predict = predict_df[predict_df['state'] == state_value]
        state_predict_yravg = state_predict.groupby('year', as_index=False).agg({feature:'mean'})
        
        title = 'Historical Trends for ' + feat_name + ' (average over all counties)'
        yaxis = state_train_yravg[feature].append(state_predict_yravg[feature])
        
        # National average
        natavg_train_county = train_df.groupby(['state', 'year'], as_index=False).agg({feature:'mean'})
        natavg_train = natavg_train_county.groupby('year', as_index=False).agg({feature:'mean'})
        
        natavg_predict_county = predict_df.groupby(['state', 'year'], as_index=False).agg({feature:'mean'})
        natavg_predict = natavg_predict_county.groupby('year', as_index=False).agg({feature:'mean'})
        natyaxis = natavg_train[feature].append(natavg_predict[feature])

    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=state_train_yravg['year'].append(state_predict_yravg['year']), 
                             y=yaxis,
                             mode='lines+markers',
                             name = state_value))
    fig.add_trace(go.Scatter(x=natavg_train['year'].append(natavg_predict['year']),
                             y = natyaxis,
                             line = dict(dash='dash'),
                             name = 'National Average'))
    
    #fig.update_layout(title=title)
    fig.update_xaxes(title='Year')
    fig.update_layout(title=feat_name.title() + ' in ' + units + '<br>' + aggby.capitalize())
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(yanchor="bottom", y=0.01,
                                  xanchor="left", x=0.01))
    return fig

    
    