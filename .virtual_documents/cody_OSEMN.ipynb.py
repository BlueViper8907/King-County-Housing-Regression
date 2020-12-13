import pandas as pd
import numpy as np
import seaborn as sns
import mlxtend


import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from math import sin, cos, sqrt, atan2, radians
from sklearn import svm
from scipy.stats import zscore
from sklearn import linear_model
from statsmodels.formula.api import ols
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from statsmodels.stats.outliers_influence import variance_inflation_factor


get_ipython().run_line_magic("matplotlib", " inline")
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#wrote up our data types to save on computer space and stop some of them from being inccorectly read as objs
kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 
             'floors': float, 'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 
             'yr_built': int, 'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float}


kc_data = pd.read_csv(r'~\Documents\Flatiron\data\data\kc_house_data.csv', parse_dates = ['date'], dtype=kc_dtypes)
schools = pd.read_csv(r'~\Documents\Flatiron\data\data\Schools.csv')


#calculate distance between schools and data 
kc = {}
kc5 = {}
# approximate radius of earth in miles  miles
i = 0
#iterate over each of our rows in the dataframe
while i <= 21480:
    R = 3963.0
    k = 0
    lat1 = radians(kc_data['lat'].iloc[i])
    lon1 = radians(kc_data['long'].iloc[i])
    distance = []
    #iterate over each school to see which school is the closest to each row in our datframe 
    while k <= 641:
        lat2 = radians(schools['LAT_CEN'].iloc[k])
        lon2 = radians(schools['LONG_CEN'].iloc[k])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance.append(R * c)
        
        k += 1 
    #sort schools by distance 
    distance.sort()
    #choose closest school 
    kc[i] = distance[0:1]
    #find some of distance to nearest 5 schools 
    kc5[i] = sum(distance[0:5])
    i += 1


kc1 = pd.DataFrame.from_dict(kc, orient='index', columns=['mi_nearest_scl'])
kc5 = pd.DataFrame.from_dict(kc5, orient='index', columns=['mi_5_scls'])


kc_data = kc_data.merge(kc1, left_index=True, right_index=True)
kc_data = kc_data.merge(kc5, left_index=True, right_index=True)


# checking to see if maybe percentile matters in terms of distance, as most of our hoems are relitively close 
kc_data['prcnt_2_scl'] = kc_data['mi_nearest_scl'].rank(pct = True) 
kc_data['prcnt_2_5_scls'] = kc_data['mi_5_scls'].rank(pct = True)


kc_data.describe()


kc_data.info()


kc_data.isnull().sum()


kc_data = kc_data.drop(['id', 'date'], 1)


#to use sqft basment later on we need to convert it to a float 
kc_data['sqft_basement'] = kc_data['sqft_basement'].replace({'?': 0})
kc_data['sqft_basement'] = kc_data['sqft_basement'].astype(dtype=float)


kc_data = kc_data.fillna(0)


#Convert to integer for whole number year, not sure why it'll let us reassign it here but raise errors in dtypes
kc_data['yr_renovated'] = kc_data['yr_renovated'].astype('int')


# fixing condition to be a good or bad, hoping that'll help get rid of the multicolinearity 
kc_data['condition'] = kc_data.condition.replace(to_replace = [1.0, 2.0, 3.0, 4.0, 5.0],  value= ['bad', 'bad', 'good', 'good', 'good'])


#we have 70 zipcodes and 120 years, it would add too much complexity to our data to increase it by 190 columns
# so instead, we're going to go through and bin them! 
zips = []
years = []


for zipcode in kc_data.zipcode:
    zips.append(zipcode)
for year in kc_data.yr_built:
    years.append(year)
    
zips = list(set(zips))
years = list(set(years))

zips.sort()
years.sort()


#will have to find a way to write this into a loop at some point, but, I can't figure out how to get .replace()
#to adequatley read lists of lists while also giving them unique names, so for now this works 
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[0:5],  value= 'zip001t005')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[5:10], value= 'zip006t011')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[10:15], value= 'zip014t024')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[15:20], value= 'zip027t031')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[20:25], value= 'zip032t039')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[25:30], value= 'zip040t053')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[30:35], value= 'zip055t065')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[35:40], value= 'zip070t077')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[40:45], value= 'zip092t106')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[45:50], value= 'zip107t115')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[50:55], value= 'zip116t122')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[55:60], value= 'zip125t144')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[60:65], value= 'zip146t168')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[65:70], value= 'zip177t199')


#gonna do the same for year built by 20 years, will give us 6 new columns, may be illuminating 
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[0:20], value= 'thru20')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[20:40], value= 'thru40')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[40:60], value= 'thru60')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[60:80], value= 'thru80')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[80:100], value= 'thru2000')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[100:120], value= 'thru2020')


# get dummies of our new variables 
dummys = ['zipcode', 'yr_built', 'condition', ]

for dummy in dummys:
    dumm = pd.get_dummies(kc_data[dummy], drop_first=True)
    kc_data = kc_data.merge(dumm, left_index=True, right_index=True)

#we're doing something unique to these variables so it wouldn't save us any time to put them into a loop
dumm = pd.get_dummies(kc_data['view'], prefix='view', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)
dumm = pd.get_dummies(kc_data['grade'], prefix='gra', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)


#break up variables into diverse ranges & renaming our dummies so that they'r easier to interpret 
kc_data = kc_data.rename({'view_1.0': 'view1', 'view_2.0': 'view2', 'view_3.0': 'view3', 'view_4.0':'view4'},axis=1)
kc_data = kc_data.rename({'gra_4': 'D', 'gra_5':'Cmin', 'gra_6':'C','gra_7':'Cpl', 'gra_8':'Bmin', 'gra_9':'B',
                          'gra_10':'Bpl', 'gra_11':'Amin', 'gra_12':'A', 'gra_13':'Apl'},axis=1)


kc_data.hist(figsize=(10,10))
plt.tight_layout()


fig = pd.plotting.scatter_matrix(kc_data,figsize=(16,16));


fig, ax = plt.subplots(figsize=(16,12))
corr = kc_data.corr().abs().round(3)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr, annot=True, mask=mask, cmap='Oranges', ax=ax)
plt.setp(ax.get_xticklabels(), 
         rotation=45, 
         ha="right",
         rotation_mode="anchor")
ax.set_title('Correlations')
fig.tight_layout()


kc_data['sqft_basement'] = kc_data['sqft_basement'].map(lambda x :  1 if x == 0 else x )


#getting rid of multicolinearity in sqftage 
kc_data['sqft_total'] = kc_data['sqft_living']*kc_data['sqft_lot']
kc_data['sqft_neighb'] = kc_data['sqft_living15']*kc_data['sqft_lot15']
kc_data['sqft_habitable'] = kc_data['sqft_above']*kc_data['sqft_basement']


#print columns we will be using going forward 
#make a copy of the dataframe holding only columns we'll be including
kc_data.columns
kc_data = kc_data[['price', 'bedrooms', 'bathrooms', 'floors','waterfront', 
                   'yr_renovated', 'lat', 'long', 
                   'sqft_total', 'sqft_neighb', 'sqft_habitable', 
                   'good', 'view1', 'view2', 'view3', 'view4', 
                   'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin', 
                   'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 
                   'zip040t053', 'zip055t065', 'zip070t077', 'zip092t106', 
                   'zip107t115', 'zip116t122', 'zip125t144', 'zip146t168', 
                   'zip177t199', 
                   'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80'
                  ]]


lowtier = kc_data[kc_data.price <=300000]
midtier = kc_data[(kc_data.price > 300001) & (kc_data.price<=800000) ]
hightier = kc_data[kc_data.price >800000]

lowincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 
          'yr_renovated', 'lat', 'long', 
          'sqft_total', 'sqft_neighb', 'sqft_habitable', 
          'good', 'view1', 'view2', 'view3', 'view4', 
          'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin',  
          'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 
          'zip040t053', 'zip055t065', 'zip070t077', 'zip092t106', 
          'zip107t115', 'zip116t122', 'zip125t144', 'zip146t168', 
          'zip177t199', 
          'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80',
          'mi_nearest_scl']

mediumincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 
          'yr_renovated', 'lat', 'long', 
          'sqft_total', 'sqft_neighb', 'sqft_habitable', 
          'good', 'view1', 'view2', 'view3', 'view4', 
          'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin',  
          'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 
          'zip040t053', 'zip055t065', 'zip070t077', 'zip092t106', 
          'zip107t115', 'zip116t122', 'zip125t144', 'zip146t168', 
          'zip177t199', 
          'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80',
          'mi_nearest_scl']

highincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 
          'yr_renovated', 'lat', 'long', 
          'sqft_total', 'sqft_neighb', 'sqft_habitable', 
          'good', 'view1', 'view2', 'view3', 'view4', 
          'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin',  
          'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 
          'zip040t053', 'zip055t065', 'zip070t077', 'zip092t106', 
          'zip107t115', 'zip116t122', 'zip125t144', 'zip146t168', 
          'zip177t199', 
          'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80',
          'mi_nearest_scl']


def make_ols(df, x_columns, drops=None, target='price', add_constant=False):
    if drops:
        drops.append(target)
        X = df.drop(columns=drops)
    else:
        X = df[x_columns]
    if add_constant:
        X = sm.add_constant(X)
    y = df[target]
    ols = sm.OLS(y, X)
    model = ols.fit()
    display(model.summary())
    fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', alpha=.05, fit=True)
    return model


price_tiers = [('low', lowtier, lowincome), 
               ('mid', midtier, mediumincome), 
               ('high', hightier, highincome)]


for name, tier, income in price_tiers:
    print(name.upper())
    make_ols(tier, income)


for col in ['price']:
    col_zscore = str(col + '_zscore')
    kc_data[col_zscore] = (kc_data[col] - kc_data[col].mean())/kc_data[col].std()
    kc_data = kc_data.loc[kc_data[col_zscore] < 2.5]
    kc_data = kc_data.loc[kc_data[col_zscore] > (-2.5)]
    kc_data = kc_data.drop(col_zscore, axis = 1)


plt.figure(figsize=(15,4))
plt.plot(kc_data['price'].value_counts().sort_index())


for i in range(1,100):
    q = i / 100
    print('{} percentile: {}'.format(q, kc_data['sqft_neighb'].quantile(q=q)))


#in bedrooms, we can clearly see a single outlier that is likely just a typo 
kc_data[kc_data['bedrooms'] == 33]
# wouldn't be realistic for a house with 33 bedrooms to only have a sqft_living of 1620 and only 1 3/4 bathrooms so we will adjust to 3 
kc_data[kc_data['bedrooms'] == 33] = kc_data[kc_data['bedrooms'] == 33].replace(33,3)


# to fix other outliers we will explore our data and find cutoffs that seem reasonable 
kc_data = kc_data.loc[kc_data['sqft_total'] <= 3.000000e+07] 
kc_data = kc_data.loc[kc_data['sqft_total'] >= 500000]
kc_data = kc_data.loc[kc_data['sqft_neighb'] <= 2.500000e+07]
kc_data = kc_data.loc[kc_data['sqft_habitable'] >= 500000]
kc_data = kc_data.loc[kc_data['sqft_habitable'] <= 2000000]
kc_data =  kc_data.loc[kc_data['bathrooms'] >= 1]
kc_data =  kc_data.loc[kc_data['bathrooms'] <= 5.5]
kc_data =  kc_data.loc[kc_data['bedrooms'] <= 7]


lowtier = kc_data[(kc_data.price > 210000) & (kc_data.price<=348000) ]
midtier = kc_data[(kc_data.price > 348000) & (kc_data.price<=480000) ]
uppermidtier = kc_data[(kc_data.price > 480000) & (kc_data.price<=664000) ]
hightier = kc_data[(kc_data.price >664000) & (kc_data.price<=1080000)]


lowincome = ['bathrooms', 'lat', 'long',  'sqft_habitable', 
             'view1', 'view2', 'view3', 'Cpl', 'Bmin', 'B', 
             'Bpl', 'Amin', 'zip055t065', 'zip092t106',
             'zip107t115', 'zip116t122', 'zip146t168']

mediumincome = ['bathrooms', 'waterfront',  'lat', 'long', 
                'view2', 'Bmin', 'B', 'Bpl', 'Amin', 
                'zip006t011', 'zip032t039', 'zip040t053', 
                'zip070t077',  'zip107t115', 'zip116t122', 
                'zip125t144', 'thru2000', 'thru60', 'thru80']

uppermedincome = ['bathrooms', 'long', 'sqft_habitable',
                  'Cpl', 'Bmin', 'B', 'Bpl',
                  'zip014t024', 'zip146t168', 
                  'zip055t065', 'zip070t077', 'zip125t144', 
                  'thru2000', 'thru2020', 'thru80']

highincome = ['bathrooms', 'floors', 'sqft_neighb', 
              'sqft_habitable', 'Amin', 'thru2020',
              'zip006t011', 'zip027t031',  'zip070t077', 
              'zip107t115', 'zip116t122', 'zip177t199']


price_tiers = [('low', lowtier, lowincome), 
               ('mid', midtier, mediumincome), 
               ('upmid', uppermidtier, uppermedincome),
               ('high', hightier, highincome)]


for name, tier, income in price_tiers:
    print(name.upper())
    make_ols(tier, income)


#make_ols(hightier, highincome)
model = make_ols(lowtier, lowincome)


# print('test R Squared: ' + str(round(r2_score(X_test, y_test), 2)))


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Bpl", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "bathrooms", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "lat", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "long", fig=fig)
plt.show()


# fig = plt.figure(figsize=(15,8))
# fig = sm.graphics.plot_regress_exog(model, "sqft_total", fig=fig)
# plt.show()
lowtier.columns


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "sqft_habitable", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Bmin", fig=fig)
plt.show()


lowtier['sqft_total'].describe()


#first step
training_data, testing_data = train_test_split(hightier, test_size=0.2)


#split columns
target = 'price'
predictive_cols = training_data.drop(target, axis=1).columns


#fit the model
predictors = '+'.join(predictive_cols)
formula = target + '~' + predictors
model = ols(formula=formula, data=training_data).fit()


# predictions
y_pred_train = model.predict(training_data[predictive_cols])
y_pred_test = model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)


#first step
training_data, testing_data = train_test_split(midtier, test_size=0.2)


#split columns
target = 'price'
predictive_cols = training_data.drop(target, axis=1).columns


#fit the model
predictors = '+'.join(predictive_cols)
formula = target + '~' + predictors
model = ols(formula=formula, data=training_data).fit()


# predictions
y_pred_train = model.predict(training_data[predictive_cols])
y_pred_test = model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)


#first step
training_data, testing_data = train_test_split(lowtier, test_size=0.2)


#split columns
target = 'price'
predictive_cols = training_data.drop(target, axis=1).columns


#fit the model
predictors = '+'.join(predictive_cols)
formula = target + '~' + predictors
model = ols(formula=formula, data=training_data).fit()


# predictions
y_pred_train = model.predict(training_data[predictive_cols])
y_pred_test = model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='grade', y='price', data=kc_data)
ax.set(title='Grade relationship on Price', 
       xlabel='Grade', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='bathrooms', y='price', data=kc_data)
ax.set(title='Bathrooms & Price', 
       xlabel='Bathrooms', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='bedrooms', y='price', data=kc_data)
ax.set(title='Bedrooms & Price', 
       xlabel='Bedrooms', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='floors', y='price', data=kc_data)
ax.set(title='Floors & Price', 
       xlabel='Floors', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='condition', y='price', data=kc_data)
ax.set(title='Condition & Price', 
       xlabel='Condition', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.regplot(x='sqft_habitable', y='price', data=kc_data)
ax.set(title='Square Habitable & Price', 
       xlabel='SqFt.', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='yr_built', y='price', data=kc_data)
ax.set(title='Year Built & Price', 
       xlabel='Year Built', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='yr_renovated', y='price', data=kc_data)
ax.set(title='Year Renovated & Price', 
       xlabel='Year', ylabel='Price')

fig.tight_layout()









