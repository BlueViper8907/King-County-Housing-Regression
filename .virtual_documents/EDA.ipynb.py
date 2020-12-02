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


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


get_ipython().run_line_magic("matplotlib", " inline")
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from math import sqrt
from scipy import stats


# read data
kc_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
              'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
              'sqft_lot15']

kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 'floors': float, 
             'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 'yr_built': int,
             'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float, 'sqft_living15': int, 'sqft_lot15': int}

kc_data = pd.read_csv('dsc-phase-2-project/data/Cleaned_Dataset.csv', parse_dates = ['date'])


kcx = kc_data[['price']]
kcy = kc_data.drop('price', axis=1)


fig_dims = (12, 8)
fig, ax = plt.subplots(figsize = fig_dims)
sns.heatmap(kc_data.corr(), ax=ax)
plt.show()


#once you hit 2 bathrooms no bearing on price 
#only houses with a bed/bath ratio near the standard suggestion hit
#the highest in price, but outside of that doesn't seem to have much 
#bearing until you get to much larger houses which should cost more
#given size but don't because no one wants a 7bd 1bth 











#fitting our model by dropping features that dont do much & transforming others into 
#useful data points through multiplying them 
#removing homes with horrible bed to bath ratios 
kc_data['ratio_bd_bth'] = kc_data['bedrooms']/kc_data['bathrooms']
kc_data['sqft_abv_and_living'] = kc_data['sqft_above']*kc_data['sqft_living']
kc_data['sqft_15'] = kc_data['sqft_living15']*kc_data['sqft_lot15']
 


kc_data = kc_data.loc[kc_data['ratio_bd_bth'] < 4]
kc_data = kc_data.drop('date', axis=1)#.drop('sqft_above', axis=1).drop('id', axis=1)
#kc_data = kc_data.drop('sqft_living', axis=1).drop('sqft_basement', axis=1)

#reassign x & y 
kcx = kc_data[['price']]
kcy = kc_data.drop('price', axis=1)


plt.scatter(kc_data['grade'], kcx, alpha=.05)


# initialize regression
reg = linear_model.LinearRegression()
# split data into 80/20 training/testing
x_train, x_test, y_train, y_test = train_test_split(kcx, kcy, test_size=0.2, random_state=42)


# train the model 
reg.fit(x_train, y_train)


# print test data predictions 
y_pred = reg.predict(x_test)
y_pred


# check accuracy with mean sq 
print(np.mean((y_pred - y_test)**2))


# Define the problem
outcome = 'price'
x_cols = ['floors', 'waterfront', 'view',  'condition', 'grade', 
          'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 
          'sqft_living15','sqft_lot15', 'sqft_abv_and_living', 'ratio_bd_bth']



# Fitting the actual model
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=kc_data).fit()
model.summary()


#remove non-significant features, basement size could be significant if we could get
#rid of the zeros, but as most people don't have basements i dont see how that'd be possible


outcome = 'price'
x_cols = ['bedrooms', 'bathrooms', 'sqft_lot', 
          'floors', 'waterfront', 'view',  'condition', 'grade', 
          'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']



from statsmodels.stats.outliers_influence import variance_inflation_factor


X = kc_data[x_cols]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
list(zip(x_cols, vif))


fig = sm.graphics.qqplot(kc_data['long_zscore'], dist=stats.norm, line='45', fit=True)


import statsmodels.api as sm
import scipy.stats as stats


kc_data['bedrooms'].sort_values(ascending=True)


kc_data.loc[8182]


kc_columns =  kc_data[['bathrooms', 'sqft_living', 'grade', 'sqft_above', 'lat', 'sqft_living15', 'sqft_abv_and_living', 'grade_zscore']].copy()
x_cols = ['bathrooms', 'sqft_living', 'grade', 'sqft_above', 'lat', 'sqft_living15', 'sqft_abv_and_living', 'grade_zscore']


plt.scatter(model.predict(kc_data), model.resid)
plt.plot(model.predict(kc_data), [0 for i in range(len(kc_columns))])


plt.scatter(kc_data['floors'], kcx)
#should we keep?


from sklearn.model_selection import train_test_split





#Trying KNN
# maybe try 60% train , 20% adj, 20% test?
cv = KFold(n_splits=10)
classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
k_pred = cross_val_predict(classifier_pipeline, kcx, kcy, cv=5)



vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for val in vals:
    features = abs(kc_data.corr())['price'][abs(kc_data.corr()['price'])>val].drop('price').index.tolist()
    
    x = kc_data.drop(columns='price')
    x = x[features]
    kcy = kc_data['price'].copy()
    
    y_pred = cross_val_predict(classifier_pipeline, x, kcy, cv=cv)
    
    print(features)
    print('RMSE:  ' + str(round(sqrt(mean_squared_error(kcy, y_pred)), 2)))
    print('R Squared: ' + str(round(r2_score(kcy, y_pred), 2)))


# Define the problem
outcome = 'price'
x_cols = ['bathrooms', 'sqft_living', 'grade', 'sqft_above', 'lat', 'sqft_living15', 'sqft_abv_and_living', 'grade_zscore']


# Fitting the actual model
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=kc_data).fit()
model.summary()


kc_data['lat'].sort_values()


for col in kc_columns:
    col_zscore = col + '_zscore'
    kc_data[col_zscore] = (kc_data[col] - kc_data[col].mean())/kc_data[col].std(ddof=0)


kc_columns = ['bathrooms', 'sqft_living', 'grade', 'sqft_above', 'lat', 'sqft_living15', 'sqft_abv_and_living', 'grade_zscore']

for column in kc_columns:
    kc_data = kc_data.loc[kc_data[column] < 3]
    kc_data = kc_data.loc[kc_data[column] > (-3)]


for col in kc_columns:
    kc_data.drop(col, 1, inplace=True)


#kc_data['bathrooms'].sort_values()



from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')


clf.fit(x_train, y_train)






