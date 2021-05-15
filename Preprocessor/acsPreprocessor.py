"""
TCS Capstone Project
American Community Survey Parser
@author: Yilun Chen
@date: 03/21/2021
"""

import pandas as pd
import numpy as np
import os

def acsPreprocessor(basePath):
    # Set-up path
    acsPath = os.path.join(basePath, 'data/ACS')
    dataPath = os.path.join(basePath, 'data')
    if not os.path.exists(os.path.join(dataPath, 'Data_with_FIPS')):
        os.makedirs(os.path.join(dataPath, 'Data_with_FIPS'))
    # Specify variables
    forms = ['subject', 'detail']
    version = ['county', 'state']
    codeBookName = 'codeBook.csv'
    raceName = ['white', 'baa', 'aian', 'asian', 'nhopi', 'other']
    yearTrainPath = os.path.join(acsPath, 'yearTrain.txt')
    yearPredictPath = os.path.join(acsPath, 'yearPredict.txt')
    yearTrain = [line.strip() for line in open(yearTrainPath, 'r')]
    yearPredict = [line.strip() for line in open(yearPredictPath, 'r')]
    
    # Step 1: Process state county fips
    for yr in yearTrain+yearPredict:
        for form in forms:
            # County dataset
            data = pd.read_csv(os.path.join(
                acsPath, 'output', '{}_county_{}.csv'.format(yr, form)
            ))
            data[['county', 'state', 'fips']] = data['Unnamed: 0'].str.split(',', expand=True)
            data[['state', 'state_temp1', 'state_temp2']] = data['state'].str.split(':', expand=True)
            data[['county', 'county_temp']] = data.county.str.split(r'County', n=-1, expand=True)
            data[['fips_temp1', 'fips_temp2']] = data.fips.str.split('>', expand=True)
            data.fips = data['fips_temp1'].str.extract('(\d+)') + data['fips_temp2'].str.extract('(\d+)')
            data = data[data.columns.drop(list(data.filter(regex='temp')))]
            data.to_csv(os.path.join(
                acsPath, 'working', '{}_county_{}.csv'.format(yr, form)
            ))
            # State dataset
            data = pd.read_csv(os.path.join(
                acsPath, 'output', '{}_state_{}.csv'.format(yr, form)
            ))
            data[['state', 'fips']] = data['Unnamed: 0'].str.split(',', expand=True)
            data[['state', 'state_temp1', 'state_temp2']] = data['state'].str.split(':', expand=True)
            data[['temp', 'fips']] = data['fips'].str.split(':', expand=True)
            data = data[data.columns.drop(list(data.filter(regex='temp')))]
            data.to_csv(os.path.join(
                acsPath, 'working', '{}_state_{}.csv'.format(yr, form)
            ))
        # Merge detail and subject datasets
        # County data
        formSubj = pd.read_csv(os.path.join(
            acsPath, 'working', '{}_county_subject.csv'.format(yr)),
            index_col=[0]
        )
        formDetail = pd.read_csv(os.path.join(
            acsPath, 'working', '{}_county_detail.csv'.format(yr)),
            index_col=[0]
        )
        data = pd.merge(
            formSubj, 
            formDetail, 
            on=['state','county','fips']
        )
        data['year'] = yr
        data.to_csv(os.path.join(
            acsPath, 'working', '{}_county.csv'.format(yr)
        ))
        # State data
        formSubj = pd.read_csv(os.path.join(
            acsPath, 'working', '{}_state_subject.csv'.format(yr)),
            index_col=[0]
        )
        formDetail = pd.read_csv(os.path.join(
            acsPath, 'working', '{}_state_detail.csv'.format(yr)),
            index_col=[0]
        )
        data = pd.merge(
            formSubj, 
            formDetail, 
            on=['state', 'fips']
        )
        data['year'] = yr
        # Save merged
        data.to_csv(os.path.join(
            acsPath, 'working', '{}_state.csv'.format(yr)
        ))
    
    # Step 2: Calculate variables for subject form
    codeBook = pd.read_csv(
        os.path.join(acsPath, codeBookName)
        )
    ## Common set 
    for yr in yearTrain+yearPredict:
        if int(yr) > 2019:
            yrLabel = '2019'
        else:
            yrLabel = yr
        codeSer = codeBook['Code_{}'.format(yrLabel)]
        for ver in version:
            data = pd.read_csv(os.path.join(
                acsPath, 'working', '{}_{}.csv'.format(yr, ver)
                ))
            data['population_all'] = data[codeSer[38]]
            data['population_65'] = data[codeSer[0]]
            data['income_median_house_65'] = data[codeSer[3]]
            data['income_median_house_all'] = data[codeSer[4]]
            data['per_poverty_all'] = data[codeSer[13]]/data[codeSer[12]]
            data['per_poverty_65'] = data[codeSer[15]]/data[codeSer[14]]
            data['per_male_all'] = data[codeSer[16]]/(data[codeSer[16]]+100)
            data['per_hear_diffi_65'] = data[codeSer[19]]/100
            data['per_vision_diffi_65'] = data[codeSer[20]]/100
            data['per_cog_diffi_65'] = data[codeSer[21]]/100
            data['per_ambu_diffi_65'] = data[codeSer[22]]/100
            data['per_selfCare_diffi_65'] = data[codeSer[23]]/100
            data['per_indeLive_diffi_65'] = data[codeSer[24]]/100
            # Race      
            for i, r in enumerate(raceName):
                data['per_oneRace_{}_all'.format(r)] = data[codeSer[31+i]]/data['population_all']
            data['per_twoRace_over_all'] = data[codeSer[37]]/data['population_all']
            # Gender
            data['per_male_65'] = np.sum(data[codeSer[87:90]], axis=1)/data['population_65']
            data.to_csv(os.path.join(
                acsPath, 'working', '{}_{}.csv'.format(yr, ver)
                ))
    yearBefore = []
    yearAfter = []
    for yr in yearTrain+yearPredict:
        if int(yr) <= 2014:
            yearBefore.append(yr)
        else:
            yearAfter.append(yr)
    # Before 2017 distinct set
    for yr in yearBefore:
        codeSer = codeBook['Code_{}'.format(yr)]
        for ver in version:
            data = pd.read_csv(os.path.join(
                acsPath, 'working', '{}_{}.csv'.format(yr, ver)
                ))
            data['per_high_65'] = data[codeSer[1]]/100
            data['per_bachelor_65'] = data[codeSer[2]]/100
            data['per_employment_65'] = (data[codeSer[5]]/100*(data[codeSer[7]]+data[codeSer[8]])/100+data[codeSer[6]]/100*(data[codeSer[9]]+data[codeSer[10]]+data[codeSer[11]])/100)*data['population_all']/data['population_65']
            data['per_veteran_65'] = (data[codeSer[25]]/100*(data[codeSer[7]]+data[codeSer[8]])/100+data[codeSer[26]]/100*(data[codeSer[9]]+data[codeSer[10]]+data[codeSer[11]])/100)*data['population_all']/data['population_65']
            data['per_stamp_house_all'] = data[codeSer[27]]/data[codeSer[28]]
            data['per_insured_65'] = (data[codeSer[29]] - data[codeSer[30]])/data[codeSer[29]]        
            data = data.loc[:, ~data.columns.str.startswith('S')]
            data = data.loc[:, ~data.columns.str.startswith('U')]
            data = data.loc[:, ~data.columns.str.startswith('B')]
            data.to_csv(os.path.join(
                acsPath, 'processed', '{}_{}.csv'.format(yr, ver)
                ))
    ## after 2017 distinct set
    for yr in yearAfter:
        if int(yr) > 2019:
            yrLabel = '2019'
        else:
            yrLabel = yr
        codeSer = codeBook['Code_{}'.format(yr)]
        for ver in version:
            data = pd.read_csv(os.path.join(
                acsPath, 'working', '{}_{}.csv'.format(yr, ver)
                ))
            data['per_high_65'] = data[codeSer[1]]/data['population_65']
            data['per_bachelor_65'] = data[codeSer[2]]/data['population_65']
            if (yr=='2015') | (yr=='2016'):
                data['per_employment_65'] = (data[codeSer[5]]/100*(data[codeSer[7]]+data[codeSer[8]])/100+data[codeSer[6]]/100*(data[codeSer[9]]+data[codeSer[10]]+data[codeSer[11]])/100)*data['population_all']/data['population_65']
            else:
                data['per_employment_65'] = (data[codeSer[5]]*(data[codeSer[7]]+data[codeSer[8]])+data[codeSer[6]]*(data[codeSer[9]]+data[codeSer[10]]+data[codeSer[11]]))/data['population_65']/100
            data['per_veteran_65'] = (data[codeSer[25]]+data[codeSer[26]])/data['population_65']
            data['per_stamp_house_all'] = data[codeSer[27]]/data[codeSer[28]]
            data['per_insured_65'] = (data[codeSer[29]] - data[codeSer[30]])/data[codeSer[29]]
            data = data.loc[:, ~data.columns.str.startswith('S')]
            data = data.loc[:, ~data.columns.str.startswith('U')]
            data = data.loc[:, ~data.columns.str.startswith('B')]
            data.to_csv(os.path.join(
                acsPath, 'processed', '{}_{}.csv'.format(yr, ver)
                ))
    
    # Step 3: Concat years
    dataCounty = []
    dataState = []
    for yr in yearTrain+yearPredict:
        dataC = pd.read_csv(
            os.path.join(acsPath, 'processed', '{}_county.csv'.format(yr)),
            index_col=[0]
            )
        dataS = pd.read_csv(
            os.path.join(acsPath, 'processed', '{}_state.csv'.format(yr)),
            index_col=[0]
            )
        dataCounty.append(dataC)
        dataState.append(dataS)
    dataCounty = pd.concat(dataCounty)
    dataState = pd.concat(dataState)

    # Step 4: Data validation
    # Employment Rate
    dataCounty.loc[(dataCounty['per_employment_65']<-9), 'per_employment_65'] = np.nan
    dataState.loc[(dataState['per_employment_65']<-9), 'per_employment_65'] = np.nan
    
    # Drop duplicates
    dataCounty.drop_duplicates(['fips', 'year'], keep='first', inplace=True)
    dataState.drop_duplicates(['fips', 'year'], keep='first', inplace=True)
    
    # Save dataset
    dataCounty.to_csv(os.path.join(dataPath, 'Data_with_FIPS', 'ACS_County_FIPS.csv'))
    dataState.to_csv(os.path.join(dataPath, 'Data_with_FIPS', 'ACS_State_FIPS.csv'))

    return

