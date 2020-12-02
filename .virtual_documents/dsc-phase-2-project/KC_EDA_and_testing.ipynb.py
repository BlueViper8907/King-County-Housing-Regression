import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy 
import sklearn
import datetime


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from math import sqrt
from scipy import stats


# read data
kc_columns = ['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
              'condition', 'grade', 'sqft_above', 'sqft_basment', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
              'sqft_lot15']

kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 'floors': float, 
             'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 'yr_built': int,
             'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float, 'sqft_living15': int, 'sqft_lot15': int}

kc_data = pd.read_csv('dsc-phase-2-project/data/Cleaned_Dataset.csv', dtype = kc_dtypes, parse_dates = ['date'])

kc_data['sqft_basement'] = kc_data['sqft_basement'].astype(dtype=float, errors='ignore')
#setting waterfront NaN values equal to the ratio of waterfront/non-waterfront properties, will want to try and narrow by zipcode 
#filling NaN with easily seperatable/changable values helpful 
kc_data['waterfront'] = kc_data['waterfront'].fillna(146/19221)
kc_data['view'] = kc_data['view'].fillna((957 + (508*2) + (330*3) + (317*4))/21534)


kc_data.info()


# model cannot handle dates? 
# cannot find a singe question mark by searching but getting errors so swapping for 0
# honestly 0 fills all pretty well, it's the mode of most of them and means 'none' for things like basements, which aren't 
# common here 
kc_data = kc_data.drop('date', axis=1).copy() 


# seperate your x and y, here I am looking at price 
kcy = kc_data['price'].to_frame()
kcx = kc_data.drop('price', axis=1)


# initialize regression
reg = linear_model.LinearRegression()


# split data into 80/20 training/testing
x_train, x_test, y_train, y_test = train_test_split(kcx, kcy, test_size=0.2, random_state=42)


# train the model 
reg.fit(x_train, y_train)


# print coefficients for each feat/column 
print(reg.coef_)


# print test data predictions 
y_pred = reg.predict(x_test)
y_pred


#print vaalues
y_test


# check accuracy with mean sq 
print(np.mean((y_pred - y_test)**2))


# check accuracy with mean sq 
print(mean_squared_error(y_test, y_pred))
# very innaccurate? 


def calc_slope(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)**2) - np.mean(xs*xs)))
    
    return m

calc_slope(y_test, y_pred)


#reassign variables to help with computer speed 
kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 'floors': float, 
             'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 'yr_built': int,
             'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float, 'sqft_living15': int, 'sqft_lot15': int}

kc_data = pd.read_csv('dsc-phase-2-project/data/Cleaned_Dataset.csv', dtype = kc_dtypes, parse_dates = ['date'])

kcy = kc_data['price'].to_frame()
kcx = kc_data.drop('price', axis=1)
kc_data.info()


#Trying KNN
# maybe try 60% train , 20% adj, 20% test?
cv = KFold(n_splits=10)
classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
k_pred = cross_val_predict(classifier_pipeline, kcx, kcy, cv=5)


print('RMSE:  ' + str(round(sqrt(mean_squared_error(kcy, k_pred)), 2)))
print('R Squared: ' + str(round(r2_score(kcy, k_pred), 2)))
print('Slope: ' + str(calc_slope(kcy, k_pred)))


kc_data.var()


fig_dims = (12, 8)
fig, ax = plt.subplots(figsize = fig_dims)
sns.heatmap(kc_data.corr(), ax=ax)
plt.show()


#reassign to save space?
# read data
kc_columns = ['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
              'condition', 'grade', 'sqft_above', 'sqft_basment', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
              'sqft_lot15']

kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 'floors': float, 
             'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 'yr_built': int,
             'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float, 'sqft_living15': int, 'sqft_lot15': int}

kc_data = pd.read_csv('dsc-phase-2-project/data/Cleaned_Dataset.csv', dtype = kc_dtypes, parse_dates = ['date'])

# new drop, hoping sqft_basement is what's giving me an error, which would make sense bc i cant assign it a dtype 
kc_data = kc_data.drop('date', axis=1).copy() 
kc_data = kc_data.drop('sqft_basement', axis=1).copy() 
kc_data = kc_data.fillna('0').copy()


kcy = kc_data['price'].to_frame()
kcx = kc_data.drop('price', axis=1)



#kc_data.loc[kc_data['price'] == 0.0]
kc_data['price'].sort_values()


abs(kc_data.corr()['price'])


abs(kc_data.corr())['price'][abs(kc_data.corr()['price'])>0.7].drop('price').index.tolist()


# rank features by correlation 
# 0.7 returning empty array, .65 best correlation we're getting 

vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65]

for val in vals:
    features = abs(kc_data.corr())['price'][abs(kc_data.corr()['price'])>val].drop('price').index.tolist()
    
    x = kc_data.drop(columns='price')
    x = x[features]
    
    y_pred = cross_val_predict(classifier_pipeline, x, kcy, cv=cv)
    
    print(features)
    print('RMSE:  ' + str(round(sqrt(mean_squared_error(kcy, y_pred)), 2)))
    print('R Squared: ' + str(round(r2_score(kcy, y_pred), 2)))


# feature selection using wrapper 
kc_data['sqft_living'] = kc_data['sqft_living'].astype('category')
dummies = pd.get_dummies(kc_data['sqft_living'])
kc_dum = kc_data.drop(columns='sqft_living').merge(dummies, left_index=True, right_index=True )


sfs1 = SFS(classifier_pipeline,
          k_features = 16,
          forward = True,
          scoring = 'neg_mean_squared_error',
          cv=cv)


sfs1.fit(kcx, kcy)


sfs1.subsets_


kc_data = pd.read_csv('dsc-phase-2-project/data/Normalized_Dataset.csv', dtype = kc_dtypes, parse_dates = ['date'])
kcy = kc_data['price_zscore'].to_frame()
kcx = kc_data.drop('price_zscore', axis=1)
y_pred = cross_val_predict(classifier_pipeline, kcx, kcy, cv=cv)
print('RMSE:  ' + str(round(sqrt(mean_squared_error(kcy, y_pred)), 2)))
print('R Squared: ' + str(round(r2_score(kcy, y_pred), 2)))


kc_data.corr()


# read data
kc_columns = ['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
              'condition', 'grade', 'sqft_above', 'sqft_basment', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
              'sqft_lot15']

kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 'floors': float, 
             'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 'yr_built': int,
             'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float, 'sqft_living15': int, 'sqft_lot15': int}

kc_data = pd.read_csv('dsc-phase-2-project/data/Cleaned_Dataset.csv', dtype = kc_dtypes, parse_dates = ['date'])



kcy = kc_data['price'].to_frame()
kcx = kc_data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
              'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
              'sqft_lot15']]


kcy.info()


reg = linear_model.LinearRegression()


x_train, x_test, y_train, y_test = train_test_split(kcx, kcy, test_size=0.2, random_state=42)


reg.fit(x_train, y_train)


print(reg.coef_)


# print test data predictions 
y_pred = reg.predict(x_test)
y_pred


#print vaalues
y_test


# check accuracy with mean sq 
print(np.mean((y_pred - y_test)**2))


# check accuracy with mean sq 
print(mean_squared_error(y_test, y_pred))
# still bad but a LOT better


#Trying KNN
# maybe try 60% train , 20% adj, 20% test?
cv = KFold(n_splits=10, random_state=0)
classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
k_pred = cross_val_predict(classifier_pipeline, kcx, kcy, cv=5)


print('RMSE:  ' + str(round(sqrt(mean_squared_error(kcy, k_pred)), 2)))
print('R Squared: ' + str(round(r2_score(kcy, k_pred), 2)))
print('Slope: ' + str(calc_slope(kcy, k_pred)))


kc_data.var()


fig_dims = (12, 8)
fig, ax = plt.subplots(figsize = fig_dims)
sns.heatmap(kc_data.corr(), ax=ax)
plt.show()


abs(kc_data.corr()['price']).sort_values(ascending=False)


# rank features by correlation 
# 0.7 returning empty array, .6 best correlation we're getting 

vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for val in vals:
    features = abs(kc_data.corr())['price'][abs(kc_data.corr()['price'])>val].drop('price').index.tolist()
    
    x = kc_data.drop(columns='price')
    x = x[features]
    
    y_pred = cross_val_predict(classifier_pipeline, x, kcy, cv=cv)
    
    print(features)
    print('RMSE:  ' + str(round(sqrt(mean_squared_error(kcy, y_pred)), 2)))
    print('R Squared: ' + str(round(r2_score(kcy, y_pred), 2)))


# feature selection using wrapper 
kc_data['sqft_living'] = kc_data['sqft_living'].astype('category')
dummies = pd.get_dummies(kc_data['sqft_living'])
kc_dum = kc_data.drop(columns='sqft_living').merge(dummies, left_index=True, right_index=True )


sfs1 = SFS(classifier_pipeline,
          k_features = 18,
          forward = True,
          scoring = 'neg_mean_squared_error',
          cv=cv)


kcx


sfs1.fit(kcx, kcy)


#hide these from view, very long 
sfs1.subsets_


# using our top 4 features to build our new model 
kcy = kc_data['price'].to_frame()
kcx = kc_data[['sqft_living', 'grade', 'zipcode', 'lat']].copy()
y_pred = cross_val_predict(classifier_pipeline, kcx, kcy, cv=cv)
print('RMSE:  ' + str(round(sqrt(mean_squared_error(kcy, y_pred)), 2)))
print('R Squared: ' + str(round(r2_score(kcy, y_pred), 2)))


kc_data[['price', 'sqft_living', 'grade', 'zipcode', 'lat']].corr()


sns.pairplot(kc_data[['price', 'sqft_living', 'grade', 'zipcode', 'lat']])


kc_data.hist()



