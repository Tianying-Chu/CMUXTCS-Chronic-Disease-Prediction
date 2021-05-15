# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:19:58 2021

@author: Tianying Chu
"""

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from dataUpdate import dataUpdate
from pipeline import pipeline
from Plot.tab1Update import tab1Update
from Plot.tab2Update import countyHistoricalTrends, countyFeaturePlot
from Plot.tab3Update import stateHistoricalTrends, stateFeaturePlot
from dashboard import decode_zips

import os

def main(base_path):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    
    app.layout = html.Div([
        # Dashboard header
        html.H1('CMU X TCS-Chronic Disease Forecasting'),
        html.H2('- Heart Disease & Stroke Hospitalization Rate Prediction',
                style={'marginLeft': 100,
                       'marginBottom': 50}),
        
        # Introduction
        dcc.Markdown('''
                     ### Introduction
                     > The dashboard predicts hospitalization of heart disease and stroke. 
                     > The outcome variable comes from [CDC Heart Atlas dataset](https://nccd.cdc.gov/DHDSPAtlas/?state=County).
                     > It represents hospitalization rate per 1,000 Medicare beneficiaries (3-year average).
                     > Features are social determinants of health, 
                     > including [American Community Survey (ACS)](https://www.census.gov/data/developers/data-sets/acs-1year.html), 
                     > [CMS - Chronic Conditions Spending and Prevalence](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Chronic-Conditions/CC_Main), 
                     > and [EPA Air Quality Index](https://aqs.epa.gov/aqsweb/airdata/download_files.html#Annual).
                     >                            
                     > Before running the model, please first select `Hospitalization Type` and
                     > `Prediction Year` to check and update data to the latest year.
                     > If the prediction year is after "2019-2021", please upload the latest CMS data.
                     >
                     > Then you can choose `Whether to Use Pre-trained Model` and `Model Type`, and run the model.
                     > You can see the prediction and model performance on the "Prediction" tab.
                     >
                     > If you are interested in the state-level analysis or county-level analysis, 
                     > please select `State` and `County` on the "State-Level Analysis" tab and "County-Level Analysis" tab.
                     > You can also choose the features you are interested in within the state/county on these two tabs.
                     > 
                     ''',
                     style={'marginLeft': 30,
                            'marginRight': 30}),
        
        # Model Configurations
        html.Div([
            # Upload latest CMS data
            html.Div([
                html.H6('Upload latest CMS data(if necessary):'),
                dcc.Upload(
                    id='upload-latest-cms',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                        ]),
                    style={
                        'width': '90%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                        },
                    accept='.zip',
                    multiple=True
                    ),
                html.Div(id='upload-message',
                         style={'font-size': '85%'})
                ],
                style = {'display': 'inline-block',
                         'width': '40%',
                         'height': '60%',
                         'verticalAlign': 'top',
                         'marginLeft': 15}),
            
            
                
            ], style={'marginBottom': 25, 'marginTop': 50}),
        
        html.Div([
            # Choose hospitalization type
            html.Div([
                html.H6('Hospitalization Type:'),
                dcc.Dropdown(
                    id='model-type',
                    options=[
                        {'label': 'Heart Disease', 'value': 'heart'},
                        {'label': 'Stroke', 'value': 'stroke'}
                    ],
                    value='heart'
                    )
                ],
                style={'width': '18%', 
                       'display': 'inline-block', 
                       'marginLeft': 20,
                       'marginRight': 15}),
            
            # Choose prediction year
            html.Div([
                html.H6('Prediction Year:'),
                dcc.Dropdown(
                    id='predict-label-year',
                    options=[
                        {'label': '2018-2020', 'value': '2018-2020'},
                        {'label': '2019-2021', 'value': '2019-2021'},
                        {'label': '2020-2022', 'value': '2020-2022'},
                        {'label': '2021-2023', 'value': '2021-2023'},
                        {'label': '2022-2024', 'value': '2022-2024'},
                        {'label': '2023-2025', 'value': '2023-2025'},
                    ],
                    value='2019-2021'
                    )
                ],
                style={'width': '15%', 
                       'display': 'inline-block', 
                       'marginRight': 15}),
            
            # Check feasibility
            html.Div([
                html.H6('Check Feasibility'),
                html.Button(id='check-years-button', children='Check Years'),
                dcc.Markdown(id='latest-year-message',
                             style={'font-size': '85%'})
                ],
                style={'display': 'inline-block', 
                       'verticalAlign': 'top',
                       'marginRight': 15}),
            ], style={'marginBottom': 25, 'marginTop': 25}),
        
        html.Div([
            # Choose whether to use pre-trained model
            html.Div([
                html.H6('Use Pre-trained Model:'),
                dcc.Dropdown(
                    id='load-model',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    value='Yes'
                    )
                ],
                style={'width': '20%', 
                       'display': 'inline-block', 
                       'marginLeft': 20,
                       'marginRight': 15}),
            
            # Choose model type
            html.Div([
                html.H6('Model Type:'),
                dcc.Dropdown(
                    id='model-name',
                    options=[
                        {'label': 'Linear Regression', 'value': 'LR'},
                        {'label': 'Random Forest', 'value': 'RF'},
                        {'label': 'Neural Networks', 'value': 'NN'},
#                        {'label': 'K Nearest Neighbors', 'value': 'KNN'},
                        {'label': 'Recurrent Neural Networks', 'value': 'RNN'}
                    ],
                    value='NN'
                    )
                ],
                style={'width': '20%', 
                       'display': 'inline-block',
                       'marginRight': 15}),
            
            # Submit button
            html.Div([
                html.H6('Run model'),
                html.Button(id='submit-button', children='Submit'),
                html.Div(id='prediction-message'),
                ],
                style = {'display': 'inline-block', 
                         'marginLeft': 15,
                         'verticalAlign': 'top'}),

            ], style={'marginTop': 30}),
        
        dcc.Tabs([
            # Tab content for prediction
            dcc.Tab(label='Prediction', children=[
                
                # Plot Map of Prediction
                html.Div([
                    html.H4('US Map of Prediction'),
                    dcc.Graph(id='us-map-of-prediction'),
                    ],
                    style={'width': '90%', 
                           'display': 'inline-block', 
                           'height': 500,
                           'marginLeft': 30,
                           'horizontalAlign': 'middle'}),
                
                html.Div([
                    # Display prediction value table
                    html.Div([
                        html.H4('Table of Prediction'),
                        html.Button(id='download-prediction-button', 
                                    children='Download csv',
                                    style={'marginBottom': 15}),
                        dcc.Download(id='download-prediction'),
                        html.Table(id='table-of-prediction')
                        ],
                        style={'width': '48%', 
                               'display': 'inline-block', 
                               'height': 700, 
                               'overflowX': 'scroll',
                               'marginLeft': 30}),
                    
                    # Display model metrics
                    html.Div([
                        html.H4('Model Metrics'),
                        html.Div(id='model-metrics'),
                        ],
                        style={'width': '48%', 
                               'display': 'inline-block',
                               'verticalAlign': 'top',
                               'height': 700,
                               'marginLeft': 15}),
                    
                    ], style={'marginTop': 30,
                              'horizontalAlign': 'middle'}),
                
                html.Div([
                    # Plot Feature Importance Plot
                    html.Div([
                        html.H4('Top 10 Most Important Features'),
                        dcc.Graph(id='feature-importance-plot'),
                        ],
                        style={'width': '48%', 
                               'display': 'inline-block',
                               'marginLeft': 30}),
                    
                    # Plot Feature contribution
                    html.Div([
                        html.H4('Feature Contribution Plot by Observations'),
                        #dcc.Graph(id='feature-contribution-plot'),
                        html.Div(id='feature-contribution-plot'),
                        ],
                        style={'width': '48%', 
                               'display': 'inline-block',
                               'verticalAlign':'top',
                               'marginLeft': 15}),
                    
                    ], style={'marginTop': 30,
                              'horizontalAlign': 'middle'}),
                ]),
            
            # Tab content for state-level analysis
            dcc.Tab(label='State-Level Analysis', children=[
                # Choose state
                html.Div([
                    # Select state
                    html.Div([
                        html.H6('State'),
                        dcc.Dropdown(id='state-state')
                        ],
                        style={'width': '24%', 
                               'display': 'inline-block'}),
                    ], style={'marginTop': 25,
                              'marginLeft': 30}),
                    

                # Plot historical trend at state level
                html.Div([
                    # Historical prediction trend
                    html.Div([
                        html.H4('Historical Prediction Trends'),
                        dcc.Graph(id='state-historical-trends')
                        ],
                        style={'width': '90%',
                               'display': 'inline-block'}),
                    ], style={'marginTop': 50, 
                              'marginLeft': 30}),
                
                html.Div([
                    # Feature trends 1
                    html.Div([
                        # Select feature 1
                        html.Div([
                            html.H6('Feature 1'),
                            html.Div([
                                html.Div(children='Data sources'),
                                dcc.Dropdown(id='state-feature-1-source'),
                                ]),
                            
                            html.Div([
                                html.Div(children='Feature name'),
                                dcc.Dropdown(id='state-feature-1',
                                             style={'font-size': '85%'}),
                                ]),
                            ],
                            style={'width': '75%', 
                                   'display': 'inline-block',
                                   'marginLeft': 50}),
                        # Plot feature 1 trends
                        dcc.Graph(id='state-feature-trends-1')
                        ],
                        style={'width': '33%',
                               'display': 'inline-block'}),
                    
                    # Feature trends 2
                    html.Div([
                        # Select feature 2
                        html.Div([
                            html.H6('Feature 2'),
                            html.Div([
                                html.Div(children='Data sources'),
                                dcc.Dropdown(id='state-feature-2-source'),
                                ]),
                            
                            html.Div([
                                html.Div(children='Feature name'),
                                dcc.Dropdown(id='state-feature-2',
                                             style={'font-size': '85%'}),
                                ]),
                            ],
                            style={'width': '75%', 
                                   'display': 'inline-block',
                                   'marginLeft': 50}),
                        # Plot feature 2 trends
                        dcc.Graph(id='state-feature-trends-2')
                        ],
                        style={'width': '33%',
                               'display': 'inline-block',
                               'marginLeft': 5}),
                    
                    # Feature trends 3
                    html.Div([
                        # Select feature 3
                        html.Div([
                            html.H6('Feature 3'),
                            html.Div([
                                html.Div(children='Data sources'),
                                dcc.Dropdown(id='state-feature-3-source'),
                                ]),
                            
                            html.Div([
                                html.Div(children='Feature name'),
                                dcc.Dropdown(id='state-feature-3',
                                             style={'font-size': '85%'}),
                                ]),
                            ],
                            style={'width': '75%', 
                                   'display': 'inline-block',
                                   'marginLeft': 50}),
                        # Plot feature 3 trends
                        dcc.Graph(id='state-feature-trends-3')
                        ],
                        style={'width': '33%',
                               'display': 'inline-block',
                               'marginLeft': 5}),
                    ])
                ]),
            
            
            # Tab content for county-level analysis
            dcc.Tab(label='County-Level Analysis', children=[
                # Choose state & county
                html.Div([
                    # Select state
                    html.Div([
                        html.H6('State'),
                        dcc.Dropdown(id='county-state')
                        ],
                        style={'width': '24%', 
                               'display': 'inline-block'}),
                    #Select county
                    html.Div([
                        html.H6('County'),
                        dcc.Dropdown(id='county-county')
                        ],
                        style={'width': '24%', 
                               'display': 'inline-block',
                               'marginLeft': 15}),
                    ], style={'marginTop': 25,
                              'marginLeft': 30}),
                
                # Plot historical trend at county level
                html.Div([
                    # Historical prediction trend
                    html.Div([
                        html.H4('Historical Prediction Trends'),
                        dcc.Graph(id='county-historical-trends')
                        ],
                        style={'width': '90%',
                               'display': 'inline-block'}),
                    ], style={'marginTop': 50, 
                              'marginLeft': 30}),
                              
                html.Div([
                    # Feature trends 1
                    html.Div([
                        # Select feature 1
                        html.Div([
                            html.H6('Feature 1'),
                            html.Div([
                                html.Div(children='Data sources'),
                                dcc.Dropdown(id='county-feature-1-source'),
                                ]),
                            
                            html.Div([
                                html.Div(children='Feature name'),
                                dcc.Dropdown(id='county-feature-1',
                                             style={'font-size': '85%'}),
                                ]),
                            ],
                            style={'width': '75%', 
                                   'display': 'inline-block',
                                   'marginLeft': 50}),
                        # Plot feature 1 trends
                        dcc.Graph(id='county-feature-trends-1')
                        ],
                        style={'width': '33%',
                               'display': 'inline-block'}),
                    
                    # Feature trends 2
                    html.Div([
                        # Select feature 2
                        html.Div([
                            html.H6('Feature 2'),
                            html.Div([
                                html.Div(children='Data sources'),
                                dcc.Dropdown(id='county-feature-2-source'),
                                ]),
                            
                            html.Div([
                                html.Div(children='Feature name'),
                                dcc.Dropdown(id='county-feature-2',
                                             style={'font-size': '85%'}),
                                ]),
                            ],
                            style={'width': '75%', 
                                   'display': 'inline-block',
                                   'marginLeft': 50}),
                        # Plot feature 2 trends
                        dcc.Graph(id='county-feature-trends-2')
                        ],
                        style={'width': '33%',
                               'display': 'inline-block',
                               'marginLeft': 5}),
                    
                    # Feature trends 3
                    html.Div([
                        # Select feature 3
                        html.Div([
                            html.H6('Feature 3'),
                            html.Div([
                                html.Div(children='Data sources'),
                                dcc.Dropdown(id='county-feature-3-source'),
                                ]),
                            
                            html.Div([
                                html.Div(children='Feature name'),
                                dcc.Dropdown(id='county-feature-3',
                                             style={'font-size': '85%'}),
                                ]),
                            ],
                            style={'width': '75%', 
                                   'display': 'inline-block',
                                   'marginLeft': 50}),
                        # Plot feature 3 trends
                        dcc.Graph(id='county-feature-trends-3')
                        ],
                        style={'width': '33%',
                               'display': 'inline-block',
                               'marginLeft': 5}),
                    ])
                ])
            ], style={'marginTop': 30}),
        
        html.Div(id='intermediate-train-json', style={'display': 'none'}),
        html.Div(id='intermediate-predict-json', style={'display': 'none'}),
        html.Div(id='intermediate-train-feature-years', style={'display': 'none'}),
        html.Div(id='intermediate-predict-feature-year', style={'display': 'none'}),
        ])
    
    # Upload latest CMS data
    @app.callback(
        Output('upload-message', 'children'),
        Input('upload-latest-cms', 'filename'),
        State('upload-latest-cms', 'contents')
        )
    def upload_cms(filename_lst, contents_lst):
        message = 'No file uploaded.'
        if filename_lst is not None:
            message = ''
            for filename, contents in zip(filename_lst, contents_lst):
                decode_zips(base_path, contents, filename)
                message += filename + ' is uploaded successfully!\n'
        return message
    
    # Update the latest data
    @app.callback(
        Output('latest-year-message', 'children'),
        Output('intermediate-train-feature-years', 'children'),
        Output('intermediate-predict-feature-year', 'children'),
        Input('check-years-button', 'n_clicks'),
        State('model-type', 'value'),
        State('predict-label-year', 'value'),
        State('upload-latest-cms', 'filename'),
        )
    def update_data(n_clicks, model_type, predict_label_year, upload_cms):
        if n_clicks is None:
            raise PreventUpdate
        else:
            latest_year_message, train_feature_years, predict_feature_year = dataUpdate(base_path, model_type, predict_label_year, upload_cms)
        return latest_year_message, train_feature_years, predict_feature_year
    
    # Run model
    @app.callback(
        Output('prediction-message', 'children'),
        Output('table-of-prediction', 'children'),
        Output('us-map-of-prediction', 'figure'),
        Output('feature-importance-plot', 'figure'),
        Output('feature-contribution-plot', 'children'),
        Output('model-metrics', 'children'),
        Output('intermediate-train-json', 'children'),
        Output('intermediate-predict-json', 'children'),
        Input('submit-button', 'n_clicks'),
        State('model-type', 'value'),
        State('predict-label-year', 'value'),
        State('intermediate-train-feature-years', 'children'),
        State('intermediate-predict-feature-year', 'children'),
        State('load-model', 'value'),
        State('model-name', 'value'),
        )
    def update_prediction(n_clicks, model_type, predict_label_year, train_feature_years, predict_feature_year, load_model, model_name):
        if n_clicks is None:
            raise PreventUpdate
        else:
            model, train_df, predict_df = pipeline(base_path, model_type, predict_label_year, train_feature_years, predict_feature_year, load_model, model_name, upload_cms)
            df_table, map_us, model_metrics, feature_contribution, feature_importance = tab1Update(base_path, predict_df, train_df, predict_label_year, model_type, model_name)
            
            if model_type == 'heart':
                message = ('Heart disease hospitalization in {} with {} model is successfully predicted!\n'
                           .format(predict_label_year, model_name))
            if model_type == 'stroke':
                message = ('Stroke hospitalization in {} with {} model is successfully predicted!\n'
                           .format(predict_label_year, model_name))
            
            train_json = train_df.to_json(date_format = 'iso', orient = 'split')
            predict_json = predict_df.to_json(date_format = 'iso', orient = 'split')
            return message, df_table, map_us, feature_importance, feature_contribution, model_metrics, train_json, predict_json
    
    # Update state dropdowns
    @app.callback(
        Output('state-state', 'options'),
        Output('county-state', 'options'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_dropdown_state(train_json, predict_json):
        if train_json is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
                
            available_state_train = train_df['state'].unique()
            available_state_predict = predict_df['state'].unique()
            available_state = list(set(available_state_train).union(set(available_state_predict)))
            available_state.sort()
            
            state_options = [{'label' :state, 'value' :state} for state in available_state]
            return state_options, state_options
    
    
    # Update county dropdown based on state
    @app.callback(
        Output('county-county', 'options'),
        Input('county-state', 'value'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_dropdown_county(state_value, train_json, predict_json):
        if state_value is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
            
            available_county_train = train_df[train_df['state'] == state_value]['county'].unique()
            available_county_predict = predict_df[predict_df['state'] == state_value]['county'].unique()
            available_county = list(set(available_county_train).union(set(available_county_predict)))
            available_county.sort()
            
            county_options = [{'label' :county, 'value' :county} for county in available_county]
            return county_options
    
    # Update state prediction trends
    @app.callback(
        Output('state-historical-trends', 'figure'),
        Input('state-state', 'value'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_state_prediction(state_value, train_json, predict_json):
        if state_value is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
            fig = stateHistoricalTrends(state_value, train_df, predict_df)
        return fig
    
    # Update county prediction trends
    @app.callback(
        Output('county-historical-trends', 'figure'),
        Input('county-state', 'value'),
        Input('county-county', 'value'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_county_prediction(state_value, county_value, train_json, predict_json):
        if county_value is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
            fig = countyHistoricalTrends(state_value, county_value, train_df, predict_df)
        return fig
    
    # Update state feature source dropdowns
    @app.callback(
        Output('state-feature-1-source', 'options'),
        Output('state-feature-2-source', 'options'),
        Output('state-feature-3-source', 'options'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_dropdown_state_feature_source(train_json, predict_json):
        if train_json is None:
            raise PreventUpdate
        else:
            available_source = ['AQI current year',
                                'AQI prev 8 years',
                                'ACS state',
                                'CMS prevalence',
                                'CMS spending']
            source_options = [{'label' :source, 'value' :source} for source in available_source]
            return source_options, source_options, source_options
    
    
    # Update county feature source dropdowns
    @app.callback(
        Output('county-feature-1-source', 'options'),
        Output('county-feature-2-source', 'options'),
        Output('county-feature-3-source', 'options'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_dropdown_county_feature_source(train_json, predict_json):
        if train_json is None:
            raise PreventUpdate
        else:
            available_source = ['AQI current year',
                                'AQI prev 8 years',
                                'ACS county',
                                'CMS prevalence',
                                'CMS spending']
            source_options = [{'label' :source, 'value' :source} for source in available_source]
            return source_options, source_options, source_options
    
    # Update state feature 1 dropdowns
    @app.callback(
        Output('state-feature-1', 'options'),
        Input('state-feature-1-source', 'value'),
        )
    def update_dropdown_state_feature_1(source):
        if source is None:
            raise PreventUpdate
        else:
            feature_df = pd.read_csv(base_path + '/Data/feature_documentation.csv')
            available_features = feature_df[(feature_df['data_source'] == source) & (feature_df['indicator'] == 'no')]
            feature_options = [{'label': feature[2], 'value': feature[0]} for feature in available_features.values]
            
            return feature_options
    
    # Update state feature 2 dropdowns
    @app.callback(
        Output('state-feature-2', 'options'),
        Input('state-feature-2-source', 'value'),
        )
    def update_dropdown_state_feature_2(source):
        if source is None:
            raise PreventUpdate
        else:
            feature_df = pd.read_csv(base_path + '/Data/feature_documentation.csv')
            available_features = feature_df[(feature_df['data_source'] == source) & (feature_df['indicator'] == 'no')]
            feature_options = [{'label': feature[2], 'value': feature[0]} for feature in available_features.values]
            return feature_options
    
    # Update state feature 3 dropdowns
    @app.callback(
        Output('state-feature-3', 'options'),
        Input('state-feature-3-source', 'value'),
        )
    def update_dropdown_state_feature_3(source):
        if source is None:
            raise PreventUpdate
        else:
            feature_df = pd.read_csv(base_path + '/Data/feature_documentation.csv')
            available_features = feature_df[(feature_df['data_source'] == source) & (feature_df['indicator'] == 'no')]
            feature_options = [{'label': feature[2], 'value': feature[0]} for feature in available_features.values]
            return feature_options
    
    # Update county feature 1 dropdowns
    @app.callback(
        Output('county-feature-1', 'options'),
        Input('county-feature-1-source', 'value'),
        )
    def update_dropdown_county_feature_1(source):
        if source is None:
            raise PreventUpdate
        else:
            feature_df = pd.read_csv(base_path + '/Data/feature_documentation.csv')
            available_features = feature_df[(feature_df['data_source'] == source) & (feature_df['indicator'] == 'no')]
            feature_options = [{'label': feature[2], 'value': feature[0]} for feature in available_features.values]
            return feature_options
    
    # Update county feature 2 dropdowns
    @app.callback(
        Output('county-feature-2', 'options'),
        Input('county-feature-2-source', 'value'),
        )
    def update_dropdown_county_feature_2(source):
        if source is None:
            raise PreventUpdate
        else:
            feature_df = pd.read_csv(base_path + '/Data/feature_documentation.csv')
            available_features = feature_df[(feature_df['data_source'] == source) & (feature_df['indicator'] == 'no')]
            feature_options = [{'label': feature[2], 'value': feature[0]} for feature in available_features.values]
            return feature_options
    
    # Update county feature 3 dropdowns
    @app.callback(
        Output('county-feature-3', 'options'),
        Input('county-feature-3-source', 'value'),
        )
    def update_dropdown_county_feature_3(source):
        if source is None:
            raise PreventUpdate
        else:
            feature_df = pd.read_csv(base_path + '/Data/feature_documentation.csv')
            available_features = feature_df[(feature_df['data_source'] == source) & (feature_df['indicator'] == 'no')]
            feature_options = [{'label': feature[2], 'value': feature[0]} for feature in available_features.values]
            return feature_options
    
    # Update state feature 1 trends
    @app.callback(
        Output('state-feature-trends-1', 'figure'),
        Input('state-state', 'value'),
        Input('state-feature-1', 'value'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_state_feature_1(state_value, feature_value, train_json, predict_json):
        if feature_value is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
            fig = stateFeaturePlot(state_value, feature_value, train_df, predict_df, base_path)
        return fig
    
    # Update state feature 2 trends
    @app.callback(
        Output('state-feature-trends-2', 'figure'),
        Input('state-state', 'value'),
        Input('state-feature-2', 'value'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_state_feature_2(state_value, feature_value, train_json, predict_json):
        if feature_value is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
            fig = stateFeaturePlot(state_value, feature_value, train_df, predict_df, base_path)
        return fig
    
    # Update state feature 3 trends
    @app.callback(
        Output('state-feature-trends-3', 'figure'),
        Input('state-state', 'value'),
        Input('state-feature-3', 'value'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_state_feature_3(state_value, feature_value, train_json, predict_json):
        if feature_value is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
            fig = stateFeaturePlot(state_value, feature_value, train_df, predict_df, base_path)
        return fig
    
    # Update county feature 1 trends
    @app.callback(
        Output('county-feature-trends-1', 'figure'),
        Input('county-state', 'value'),
        Input('county-county', 'value'),
        Input('county-feature-1', 'value'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_county_feature_1(state_value, county_value, feature_value, train_json, predict_json):
        if feature_value is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
            fig = countyFeaturePlot(state_value, county_value, feature_value, train_df, predict_df, base_path)
        return fig
    
    # Update county feature 2 trends
    @app.callback(
        Output('county-feature-trends-2', 'figure'),
        Input('county-state', 'value'),
        Input('county-county', 'value'),
        Input('county-feature-2', 'value'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_county_feature_2(state_value, county_value, feature_value, train_json, predict_json):
        if feature_value is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
            fig = countyFeaturePlot(state_value, county_value, feature_value, train_df, predict_df, base_path)
        return fig
    
    # Update county feature 3 trends
    @app.callback(
        Output('county-feature-trends-3', 'figure'),
        Input('county-state', 'value'),
        Input('county-county', 'value'),
        Input('county-feature-3', 'value'),
        Input('intermediate-train-json', 'children'),
        Input('intermediate-predict-json', 'children'),
        )
    def update_county_feature_3(state_value, county_value, feature_value, train_json, predict_json):
        if feature_value is None:
            raise PreventUpdate
        else:
            train_df = pd.read_json(train_json, orient='split')
            predict_df = pd.read_json(predict_json, orient='split')
            fig = countyFeaturePlot(state_value, county_value, feature_value, train_df, predict_df, base_path)
        return fig
    
    # Download prediction csv
    @app.callback(
        Output('download-prediction', 'data'),
        Input('download-prediction-button', 'n_clicks'),
        State('model-type', 'value'),
        State('predict-label-year', 'value'),
        State('load-model', 'value'),
        State('model-name', 'value'),
        prevent_initial_call=True,
        )
    def download_prediction(n_clicks, model_type, predict_label_year, load_model, model_name):
        if n_clicks is None:
            raise PreventUpdate
        else:
            df = pd.read_csv(base_path + '/Result/Prediction/Prediction_{}_{}_{}.csv'.format(model_type, model_name,  predict_label_year))
            return dcc.send_data_frame(df.to_csv, 'Prediction_{}_{}_{}.csv'.format(model_type, model_name, predict_label_year))
    
    app.run_server(debug=True)

if __name__ == '__main__':
    base_path = os.getcwd().replace('\\', '/')
    main(base_path)