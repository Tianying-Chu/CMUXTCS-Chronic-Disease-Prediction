"""
TCS Capstone Project
American Community Survey Parser
@author: Yilun Chen
@date: 03/21/2021
"""

import pandas as pd
import numpy as np
import os
import censusdata

def acsScraper(yearTrain, yearPredict, basePath):
    # Set up folder path
    dataPath = os.path.join(basePath, 'data/ACS')
    if not os.path.exists(os.path.join(dataPath, 'output')):
        os.makedirs(os.path.join(dataPath, 'output'))
    if not os.path.exists(os.path.join(dataPath, 'processed')):
        os.makedirs(os.path.join(dataPath, 'processed'))
    if not os.path.exists(os.path.join(dataPath, 'working')):
        os.makedirs(os.path.join(dataPath, 'working')) 
    # Wrie year files
    with open(os.path.join(dataPath, 'yearTrain.txt'), 'w') as file:
        file.writelines("%s\n" % year for year in yearTrain)
    with open(os.path.join(dataPath, 'yearPredict.txt'), 'w') as file:
        file.writelines("%s\n" % year for year in yearPredict)
    # Read in codes for variables
    codeBookName = 'codeBook.csv'
    codeBook = pd.read_csv(os.path.join(dataPath, codeBookName))
    trainTestInd = 31
    # Specify variables
    key = 'fd1a0d22a6d77b76707e2e9d67ae81762f8abd65'
    version = ['county', 'state']
    year_no = []
    # Loop over years
    for yr in yearTrain+yearPredict:
        if int(yr) > 2019:
            yrLabel = '2019'
        else:
            yrLabel = yr
        varSubj = list(codeBook['Code_{}'.format(yrLabel)][:trainTestInd])
        varDetail = list(codeBook['Code_{}'.format(yrLabel)][trainTestInd:])
        # Get both subject and detail tables
        for ver in version:
            try:
                dataSubj = censusdata.download(
                    src='acs1', 
                    year=int(yr),
                    geo=censusdata.censusgeo([(ver, '*')]),
                    var=varSubj, 
                    key=key, 
                    tabletype='subject'
                )
                dataDetail = censusdata.download(
                    src='acs1', 
                    year=int(yr),
                    geo=censusdata.censusgeo([(ver, '*')]),
                    var=varDetail, 
                    key=key, 
                    tabletype='detail'
                )
            except:
                year_no.append(yr)
            # Save data files
            dataSubj.to_csv(os.path.join(dataPath, 'output', '{}_{}_subject.csv'.format(yr, ver)))
            dataDetail.to_csv(os.path.join(dataPath, 'output', '{}_{}_detail.csv'.format(yr, ver)))
    if len(year_no) == 0:
        latest_year = int(yr)
    else:
        latest_year = int(year_no[0])-1
    return latest_year
    