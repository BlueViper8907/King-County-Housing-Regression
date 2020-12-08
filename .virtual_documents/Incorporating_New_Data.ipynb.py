import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import sklearn 
import math 
import scipy
import datetime 
import mlxtend
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor


import pandas as pd 
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
import os
get_ipython().run_line_magic("matplotlib", " inline")
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from math import sqrt
from scipy import stats


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


pd.options.display.max_rows = 4000


kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 'floors': float, 
             'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 'yr_built': int,
             'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float, 'sqft_living15': int, 'sqft_lot15': int}


flatiron_data = pd.read_csv(r'~\Documents\Flatiron\data\data\Cleaned_Dataset.csv', parse_dates = ['date'], dtype=kc_dtypes)


parcel_dtypes = {'Major' : int, 'Minor' : int, 'PlatLot' : str, 'PlatBlock' : str, 'Range' : int, 'Township' : int, 'Section' : int,
                 'QuarterSection' : str, 'PropType' : str, 'DistrictName' : str , 'CurrentZoning':str, 'Unbuildable': int,  'WfntAccessRights' :str, 
                 'WfntProximityInfluence':str, 'PowerLines': str, 'OtherNuisances' : str ,  'DNRLease': str, 'AdjacentGolfFairway':str, 
                 'AdjacentGreenbelt':str, 'Easements' :str, 'OtherDesignation':str, 'DeedRestrictions':str, 'DevelopmentRightsPurch': str, 
                 'CoalMineHazard': str, 'CriticalDrainage':str, 'ErosionHazard':str, 'LandfillBuffer':str, 'HundredYrFloodPlain':str,
                 'SeismicHazard': str, 'LandslideHazard':str, 'SteepSlopeHazard':str, 'Stream':str, 'Wetland':str, 'SpeciesOfConcern':str,
                 'SensitiveAreaTract':str, 'WaterProblems':str, 'TranspConcurrency': str, 'OtherProblems':str, 'id': int }


kc_parcels = pd.read_csv(r'~\Documents\Flatiron\data\data\EXTR_Parcel.csv', encoding='ISO-8859-1', dtype=parcel_dtypes)
kc_resident = pd.read_csv(r'~\Documents\Flatiron\data\data\EXTR_ResBldg.csv', encoding='ISO-8859-1', low_memory=False)


#kc_condos1 = pd.read_csv(r'~\Documents\Flatiron\data\data\EXTR_CondoComplex.csv', encoding='ISO-8859-1', low_memory=False)
#kc_condos2 = pd.read_csv(r'~\Documents\Flatiron\data\data\EXTR_CondoUnit2.csv', encoding='ISO-8859-1', low_memory=False)


kc_appraisal = pd.read_csv(r'~\Documents\Flatiron\data\data\EXTR_RealPropApplHist_V.csv', encoding='ISO-8859-1', low_memory=False)
#kc_breakdown = pd.read_csv(r'~\Documents\Flatiron\data\data\EXTR_UnitBreakdown.csv', encoding='ISO-8859-1', low_memory=False)


kc_resident = kc_resident.drop('ZipCode', axis=1).drop('DirectionSuffix', axis=1).drop('DirectionPrefix', axis=1)
kc_parcels = kc_parcels.drop('PlatName', axis=1).drop('PropName', axis=1).drop('SpecArea', axis =1).drop('SpecSubArea',  axis=1)


#kc_data.columns = kc_data.columns.droplevel(0)
#flatiron_data.columns = pd.MultiIndex.from_product([flatiron_data.columns, ['FI']])
#kc_appraisal.columns = pd.MultiIndex.from_product([kc_appraisal.columns, ['APP']])

kc_parcels = kc_parcels[['SqFtLot',
       'Access', 'Topography', 'StreetSurface', 'RestrictiveSzShape',
       'InadequateParking', 'PcntUnusable', 'Unbuildable', 'MtRainier',
       'Olympics', 'Cascades', 'Territorial', 'SeattleSkyline', 'PugetSound',
       'LakeWashington', 'LakeSammamish', 'SmallLakeRiverCreek', 'OtherView',
       'WfntLocation', 'WfntFootage', 'WfntPoorQuality',
       'WfntAccessRights', 'WfntProximityInfluence',
       'TidelandShoreland', 'LotDepthFactor', 'TrafficNoise', 'AirportNoise',
       'PowerLines', 'OtherNuisances', 'NbrBldgSites', 'Contamination',
       'DNRLease', 'AdjacentGreenbelt', 'HistoricSite',
       'CurrentUseDesignation', 'NativeGrowthProtEsmt', 'Easements',
       'OtherDesignation', 'DeedRestrictions', 'DevelopmentRightsPurch',
       'CoalMineHazard', 'CriticalDrainage', 'ErosionHazard', 'LandfillBuffer',
       'HundredYrFloodPlain', 'LandslideHazard',
       'SteepSlopeHazard', 'Stream', 'Wetland',
       'SensitiveAreaTract', 'WaterProblems', 
       'OtherProblems', 'id']].copy()

#kc_resident = kc_resident[['BldgNbr', 'NbrLivingUnits', 'Address',
#       'BuildingNumber', 'Fraction', 'StreetName', 'StreetType', 'Stories',
#       'BldgGrade', 'BldgGradeVar', 'SqFt1stFloor', 'SqFtHalfFloor',
#       'SqFt2ndFloor', 'SqFtUpperFloor', 'SqFtUnfinFull', 'SqFtUnfinHalf',
#       'SqFtTotLiving', 'SqFtTotBasement', 'SqFtFinBasement',
#       'FinBasementGrade', 'SqFtGarageBasement', 'SqFtGarageAttached',
#       'DaylightBasement', 'SqFtOpenPorch', 'SqFtEnclosedPorch', 'SqFtDeck',
#       'HeatSystem', 'HeatSource', 'BrickStone', 'ViewUtilization', 'Bedrooms',
#       'BathHalfCount', 'Bath3qtrCount', 'BathFullCount', 'FpSingleStory',
#       'FpMultiStory', 'FpFreestanding', 'FpAdditional', 'YrBuilt',
#       'YrRenovated', 'PcntComplete', 'Obsolescence', 'PcntNetCondition',
 #      'Condition', 'AddnlCost', 'id']].copy()


kc_appraisal = kc_appraisal[['RollYr', 'RevalOrMaint', 'LandVal', 'ImpsVal',
       'NewDollars', 'id']].copy()

#kc_breakdown = kc_breakdown[[ 'SqFt', 'NbrBedrooms', 'NbrBaths', 'id']].copy()



kc_resident.dtypes


#kc_condos1['Minor'] = '0000'
#kc_condos2['Minor'] = '0000'


kc_web_df = [kc_parcels, kc_appraisal, kc_resident]#, kc_condos1, kc_condos2  kc_resident, kc_breakdown, 


for dataframe in kc_web_df:
    
    i = 0
    j = 0
    
    dataframe['Major'] = dataframe['Major'].astype(str)
    dataframe['Major'] = dataframe['Major'].str.strip()    
    dataframe['Minor'] = dataframe['Minor'].astype(str)
    dataframe['Minor'] = dataframe['Minor'].str.strip()
    
    for row in dataframe:
        while len(str(dataframe['Minor'][i])) + j < 4:
            dataframe['Major'][i]*10
            j += 1
            
            
    dataframe['id'] = dataframe['Major'] + dataframe['Minor']
    dataframe = dataframe.convert_dtypes(convert_integer=False)
   
    i += 1


#kc_appraisal.columns = pd.MultiIndex.from_product([kc_appraisal.columns, ['APP']])

kc_parcels = kc_parcels[['id', 'SqFtLot', 'InadequateParking']].copy()

kc_resident = kc_resident[['BldgNbr', 'NbrLivingUnits', 'Address',
       'BuildingNumber', 'Fraction', 'StreetName', 'StreetType', 'Stories',
       'BldgGrade', 'SqFt1stFloor', 'SqFtHalfFloor',
       'SqFt2ndFloor', 'SqFtUpperFloor', 'SqFtUnfinFull', 'SqFtUnfinHalf',
       'SqFtTotLiving', 'SqFtTotBasement', 'SqFtFinBasement',
       'FinBasementGrade', 'SqFtGarageBasement', 'SqFtGarageAttached',
       'DaylightBasement', 'SqFtOpenPorch', 'SqFtEnclosedPorch', 'SqFtDeck',
       'HeatSystem', 'HeatSource', 'BrickStone', 'ViewUtilization', 'Bedrooms',
       'BathHalfCount', 'Bath3qtrCount', 'BathFullCount', 'FpSingleStory',
       'FpMultiStory', 'FpFreestanding', 'FpAdditional', 'YrBuilt',
       'YrRenovated', 'PcntComplete', 'Obsolescence', 'PcntNetCondition',
       'Condition', 'AddnlCost', 'id']].copy()


kc_appraisal = kc_appraisal[['id', 'RollYr', 'LandVal', 'ImpsVal',
       'NewDollars']].copy()

#kc_breakdown = kc_breakdown[[ 'SqFt', 'NbrBedrooms', 'NbrBaths', 'id']].copy()



print(kc_parcels.columns)


kc_appraisal = kc_appraisal.loc[kc_appraisal['RollYr'] == 2021].copy()
kc_appraisal = kc_appraisal.drop('RollYr', axis =1)


kc_data = kc_appraisal.merge(kc_parcels, how='outer', on='id')
kc_data = kc_data.merge(kc_resident, how='outer', on='id')
kc_data = kc_data.replace({'?': 0, '': 0, ' ': 0, 'N': 0, 'Y': 1})


kc_data['AppraisedVal'] =  kc_data['LandVal'] + kc_data['ImpsVal'] + kc_data['NewDollars']


kc_data.columns


# Define the problem
outcome = 'AppraisedVal'
x_cols = ['LandVal', 'ImpsVal', 'NewDollars', 'SqFtLot', 'InadequateParking',
       'BldgNbr', 'NbrLivingUnits', 'Address',
       'BuildingNumber', 'Fraction', 'StreetName', 'StreetType', 'Stories',
       'BldgGrade', 'SqFt1stFloor', 'SqFtHalfFloor',
       'SqFt2ndFloor', 'SqFtUpperFloor', 'SqFtUnfinFull', 'SqFtUnfinHalf',
       'SqFtTotLiving', 'SqFtTotBasement', 'SqFtFinBasement',
       'FinBasementGrade', 'SqFtGarageBasement', 'SqFtGarageAttached',
       'DaylightBasement', 'SqFtOpenPorch', 'SqFtEnclosedPorch', 'SqFtDeck',
       'HeatSystem', 'HeatSource', 'BrickStone', 'ViewUtilization', 'Bedrooms',
       'BathHalfCount', 'Bath3qtrCount', 'BathFullCount', 'FpSingleStory',
       'FpMultiStory', 'FpFreestanding', 'FpAdditional', 'YrBuilt',
       'YrRenovated', 'PcntComplete', 'Obsolescence', 'PcntNetCondition',
       'Condition', 'AddnlCost']


kc_columns = ['LandVal', 'ImpsVal', 'NewDollars', 'SqFtLot', 'InadequateParking',
       'BldgNbr', 'NbrLivingUnits', 'Address',
       'BuildingNumber', 'Fraction', 'StreetName', 'StreetType', 'Stories',
       'BldgGrade', 'SqFt1stFloor', 'SqFtHalfFloor',
       'SqFt2ndFloor', 'SqFtUpperFloor', 'SqFtUnfinFull', 'SqFtUnfinHalf',
       'SqFtTotLiving', 'SqFtTotBasement', 'SqFtFinBasement',
       'FinBasementGrade', 'SqFtGarageBasement', 'SqFtGarageAttached',
       'DaylightBasement', 'SqFtOpenPorch', 'SqFtEnclosedPorch', 'SqFtDeck',
       'HeatSystem', 'HeatSource', 'BrickStone', 'ViewUtilization', 'Bedrooms',
       'BathHalfCount', 'Bath3qtrCount', 'BathFullCount', 'FpSingleStory',
       'FpMultiStory', 'FpFreestanding', 'FpAdditional', 'YrBuilt',
       'YrRenovated', 'PcntComplete', 'Obsolescence', 'PcntNetCondition',
       'Condition', 'AddnlCost']


zcolumns = []

   
for column in kc_data:
    if kc_data[column].dtypes == int: 
        if kc_data[column].sum() == 0:
            kc_data = kc_data.drop(column, axis=1)
    elif  kc_data[column].dtypes == float: 
        if kc_data[column].sum() == 0.0:
            kc_data = kc_data.drop(column, axis=1)


#kc_data['View'] = kc_data['MtRainier'] + kc_data['Olympics'] + 


kc_data = kc_data.dropna(axis=0)
kc_data.columns.dtypes


for col in kc_columns:
    col_zscore = str(col + '_zscore')
    kc_data[col_zscore] = (kc_data[col] - kc_data[col].mean())/kc_data[col].std()
    kc_data = kc_data.loc[kc_data[col_zscore] < 2]
    kc_data = kc_data.loc[kc_data[col_zscore] > (-2)]
    kc_data = kc_data.drop(col_zscore, axis = 1)
    zcolumns.append(col_zscore)


zcolumns


# Fitting the actual model
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=kc_data).fit()
model.summary()


#kc_data = pd.concat(objs=[kc_appraisal, kc_parcels], join='outer', axis=1, keys='id')
#kc_data.columns = kc_data.columns.droplevel(0)
#kc_data = pd.concat(objs=[kc_resident, kc_data], join='outer', axis=1, keys='id')
#kc_data.columns = kc_data.columns.droplevel(0)
#kc_data = pd.concat(objs=[flatiron_data, kc_data], join='outer', axis=1, keys='id')
#kc_data.columns = kc_data.columns.droplevel(0)
#kc_data = pd.concat(objs=[kc_breakdown, kc_data], join='outer', axis=1, keys='id')
#kc_data.columns = kc_data.columns.droplevel(0)
#kc_data = kc_data.replace({'?': 0, '': 0, ' ': 0, 'N': 0, 'Y': 1})


fig_dims = (24, 16)
fig, ax = plt.subplots(figsize = fig_dims)
sns.heatmap(kc_data.corr(), ax=ax)
plt.show()



