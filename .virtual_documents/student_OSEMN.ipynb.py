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
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import zscore
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn import svm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor


#editing our settings so we can view more of our data at once 
get_ipython().run_line_magic("matplotlib", " inline")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#wrote up our data types to save on computer space and stop some of them from being inccorectly read as objs
kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 
             'floors': float, 'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 
             'yr_built': int, 'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float}


#read csv, parsing dates and using our dtypes 
kc_data = pd.read_csv(r'~\Documents\Flatiron\data\data\kc_house_data.csv', parse_dates = ['date'], dtype=kc_dtypes)


#have some errors in sqft_basement but I want to use it, so I've gotta replace the str, fill the null values 
# & add one sqft to each basement with no sqft, so that when we multiply to get rid of multi-colinieartiy 
#we don't end up multiplying by a bunch of zeroes and wrecking stuff 
kc_data['sqft_basement'] = kc_data['sqft_basement'].replace({'?': 0})
kc_data['sqft_basement'] = kc_data['sqft_basement'].fillna(0)
kc_data['sqft_basement'] = kc_data['sqft_basement'].astype(dtype=float)

i = 0 
for sqft in kc_data['sqft_basement']:
    if kc_data['sqft_basement'].iloc[i] == 0.0:
        kc_data['sqft_basement'].iloc[i] + 1.0


#getting rid of multicolinearity in sqftage 
kc_data['sqft_total'] = kc_data['sqft_living']*kc_data['sqft_lot']
kc_data['sqft_neighb'] = kc_data['sqft_living15']*kc_data['sqft_lot15']
kc_data['sqft_habitable'] = kc_data['sqft_above']*kc_data['sqft_basement']


#in our search for outliers we found some data that was likely just a typo, let's fix that 
kc_data[kc_data['bedrooms'] == 33] = kc_data[kc_data['bedrooms'] == 33].replace(33,3)


#setting waterfront NaN values equal to the ratio of waterfront/non-waterfront properties, will want to try and narrow by zipcode 
#filling NaN with easily seperatable/changable values helpful 
kc_data = kc_data.fillna(0)


#Convert to integer for whole number year, not sure why it'll let us reassign it here but raise errors in dtypes
kc_data['yr_renovated'] = kc_data['yr_renovated'].astype('int')
# fixing condition to be a good or bad, hoping that'll help get rid of the multicolinearity 
kc_data['condition'] = kc_data.condition.replace(to_replace = [1.0, 2.0, 3.0, 4.0, 5.0],  value= ['bad', 'bad', 'good', 'good', 'good'])


#making dummies of our catagorical variables 
dumm = pd.get_dummies(kc_data['condition'], drop_first=True)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)
dumm = pd.get_dummies(kc_data['view'], prefix='view', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)
dumm = pd.get_dummies(kc_data['grade'], prefix='gra', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)


# renaming our dummies so that they'r easier to interpret 
kc_data = kc_data.rename({'view_1.0': 'view1', 'view_2.0': 'view2', 'view_3.0': 'view3', 'view_4.0':'view4'},axis=1)
kc_data = kc_data.rename({'gra_4': 'D', 'gra_5':'Cmin', 'gra_6':'C','gra_7':'Cpl', 'gra_8':'Bmin', 'gra_9':'B',
                          'gra_10':'Bpl', 'gra_11':'Amin', 'gra_12':'A', 'gra_13':'Apl'},axis=1)


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
dumm = pd.get_dummies(kc_data['zipcode'], prefix=None, drop_first=True)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)
dumm = pd.get_dummies(kc_data['yr_built'], prefix=None, drop_first=True)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)


#sqrt trasnforming price so that it's more normalized 
kc_data['pricesqrt'] = kc_data['price'].apply(np.sqrt)


print(sqrt(17728447450.16226 ))
print(sqrt(19094035777.885963))


#some of these homes don't have a full bathroom, as we're focusing on homes that would be sold to families
#wanting at least one full bathroom is a reasonable requirement. We also have a handful of homes with too 
#many bedrooms, while I'm sure some families want 10 bedrooms, 7 is probably a good stopping point there 
kc_data =  kc_data.loc[kc_data['bathrooms'] >= 1]
kc_data =  kc_data.loc[kc_data['bedrooms'] <= 7]
#drop unnessecary columns, print columns we will be using going forward 
to_drop = ['sqft_living','sqft_lot','id','date','sqft_above','sqft_basement', 'yr_built', 'condition', 'grade',
           'zipcode']
kc_data = kc_data.drop(labels=to_drop,axis=1)
kc_data.columns


kc_data = kc_data[['pricesqrt', 'price', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'yr_renovated',
                   'lat', 'long', 'sqft_neighb', 'sqft_total', 'sqft_habitable', 'good',
                   'view1', 'view2', 'view3', 'view4', 
                   'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin', 'A', 'Apl', 
                   'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 'zip040t053', 
                   'zip055t065', 'zip070t077', 'zip092t106', 'zip107t115', 'zip116t122', 
                   'zip125t144', 'zip146t168', 'zip177t199', 'thru2000', 
                   'thru2020', 'thru40', 'thru60', 'thru80']].copy()


kc_data.describe()


schools = pd.read_csv(r'~\Documents\Flatiron\data\data\Schools.csv')
schools.info()


type(schools['LAT_CEN'].iloc[2])


#calculate distance between schools and data 
kc = {}
kc5 = {}
# approximate radius of earth in miles  miles
i = 0
#iterate over each of our rows in the dataframe
while i <= 21498:
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


kc_data.isna().sum()


for col in kc_data.columns:
    try:
        print(col, kc_data[col].value_counts()[:5])
    except:
        print(col, kc_data[col].value_counts())
    print('\n')





kc_data.describe().round(3)


kc_data.hist(figsize=(10,10))
plt.tight_layout()


#fig = pd.plotting.scatter_matrix(kc_data,figsize=(30,30));
#print(type(fig))


fig, ax = plt.subplots(figsize=(26,12))

corr = kc_data.corr().abs().round(3)

mask = np.triu(np.ones_like(corr, dtype=np.bool))

sns.heatmap(corr, annot=True, mask=mask, cmap='Oranges', ax=ax)
plt.setp(ax.get_xticklabels(), 
         rotation=45, 
         ha="right",
         rotation_mode="anchor")
ax.set_title('Correlations')


outcome = 'pricesqrt'
x_cols = [ 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'yr_renovated', 
          'lat', 'long', 'sqft_neighb', 'sqft_total', 'sqft_habitable', 
          'good', 'view1', 'view2', 'view3', 'view4',
          'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin', 'A', 'Apl', 
          'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 
          'zip055t065', 'zip070t077', 'zip092t106', 
          'zip107t115', 'zip116t122', 'zip125t144', 'zip146t168', 
          'zip177t199', 
          'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80']


predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=kc_data).fit()
model.summary()


model.params.sort_values()


lowtier = kc_data[kc_data.price <=300000]
midtier = kc_data[(kc_data.price > 300001) & (kc_data.price<=800000) ]
hightier = kc_data[(kc_data.price > 600000)]

lowincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront','yr_renovated',
           'lat', 'long', 'sqft_neighb', 'sqft_total', 'sqft_habitable', 'good', 
           'view1', 'view2', 'view3', 'view4', 
           'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin', 'A', 'Apl', 
           'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 'zip040t053', 
           'zip055t065', 'zip070t077', 'zip092t106', 'zip107t115', 'zip116t122', 
           'zip125t144', 'zip146t168', 'zip177t199', 'thru2000', 
           'thru2020', 'thru40', 'thru60', 'thru80']

mediumincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'yr_renovated',
           'lat', 'long', 'sqft_neighb', 'sqft_total', 'sqft_habitable',  'good', 
           'view1', 'view2', 'view3', 'view4', 
           'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin', 'A', 'Apl', 
           'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 'zip040t053', 
           'zip055t065', 'zip070t077', 'zip092t106', 'zip107t115', 'zip116t122', 
           'zip125t144', 'zip146t168', 'zip177t199', 'thru2000', 
           'thru2020', 'thru40', 'thru60', 'thru80']

highincome = ['bd_bth_ratio', 'floors', 'waterfront',
           'lattrans', 'sqft_neighb', 'sqft_total', 'sqft_habitable',  
           'view1', 'view2', 'view3', 'view4', 
           'zip014t024', 'zip027t031', 'zip055t065', 'zip092t106', 'zip116t122', 
           'zip125t144', 'zip146t168', 'zip177t199',
           'thru40', 'thru60', 'mi_nearest_scl']

def make_ols(df, x_columns, drops=None, target='pricesqrt', add_constant=False):
    if drops:
        drops.append(target)
        X = df.drop(columns=drops)
    else:
        X = df[x_columns]
    if add_constant:
        X = sm.add_constant(X)
    y = df[target]
    ols = sm.OLS(y, X)
    res = ols.fit()
    display(res.summary())
    fig = sm.graphics.qqplot(res.resid, dist=stats.norm, line='45', fit=True)
    return res
#make_ols(lowtier,lowincome)
#make_ols(midtier,mediumincome)
make_ols(hightier,highincome)
X = hightier[highincome]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
list(zip(highincome, vif))


kc_columns = ['price']


for col in kc_columns:
    col_zscore = str(col + '_zscore')
    kc_data[col_zscore] = (kc_data[col] - kc_data[col].mean())/kc_data[col].std()
    kc_data = kc_data.loc[kc_data[col_zscore] < 2.25]
    kc_data = kc_data.loc[kc_data[col_zscore] > (-2.25)]
    kc_data = kc_data.drop(col_zscore, axis = 1)


plt.figure(figsize=(15,4))
plt.plot(kc_data['price'].value_counts().sort_index())


for i in range(1,100):
    q = i / 100
    print('{} percentile: {}'.format(q, kc_data['price'].quantile(q=q)))


kc_data['flr'] = kc_data['floors']-kc_data['floors'].mean()
kc_data['bth*bd'] = kc_data['bathrooms']*kc_data['bedrooms']
kc_data['near_scl_comp'] = kc_data['mi_nearest_scl']-kc_data['mi_nearest_scl'].mean()
kc_data['near_5_scl_comp'] = kc_data['mi_5_scls']-kc_data['mi_5_scls'].mean()
kc_data['lattrans'] = kc_data['lat']-kc_data['lat'].mean()
kc_data.columns


#test vifs 
test = kc_data[['waterfront', 'yr_renovated', 'lat', 'long', 'sqft_neighb', 'sqft_total', 'sqft_habitable', 'good', 'view1', 'view2', 'view3', 'view4', 'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin', 'A', 'Apl', 'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 'zip040t053', 'zip055t065', 'zip070t077', 'zip092t106', 'zip107t115', 'zip116t122', 'zip125t144', 'zip146t168', 'zip177t199', 'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80', 'lattrans', 'longtrans', 'bd', 'bth', 'flr', 'nearest_scl_comp', 'near_5_scl_comp']]
test_cols = ['waterfront', 'yr_renovated', 'lat', 'long', 'sqft_neighb', 'sqft_total', 'sqft_habitable', 'good', 'view1', 'view2', 'view3', 'view4', 'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin', 'A', 'Apl', 'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 'zip040t053', 'zip055t065', 'zip070t077', 'zip092t106', 'zip107t115', 'zip116t122', 'zip125t144', 'zip146t168', 'zip177t199', 'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80', 'lattrans', 'longtrans', 'bd', 'bth', 'flr', 'nearest_scl_comp']
X = test[test_cols]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
list(zip(test_cols, vif))


lowtier = kc_data[(kc_data.price > 210000) & (kc_data.price<=348000) ]
midtier = kc_data[(kc_data.price > 348000) & (kc_data.price<=480000) ]
uppermidtier = kc_data[(kc_data.price > 480000) & (kc_data.price<=664000) ]
hightier = kc_data[(kc_data.price >664000) & (kc_data.price<=1080000)]

lowincome = ['sqft_neighb', 'sqft_total', 'sqft_habitable', 
             'view3', 'zip006t011', 
             'zip014t024', 'zip027t031', 'zip032t039', 'zip040t053', 'zip055t065',
             'zip070t077', 'zip092t106', 'zip107t115', 'zip116t122', 'zip125t144', 
             'zip146t168', 'zip177t199', 'thru2000', 'thru2020', 'thru40', 'thru60', 
             'thru80', 'lattrans', 'longtrans', 'bth*bd', 'flr', 'nearest_scl_comp']


mediumincome = ['bathrooms', 'lat', 'long', 
                'sqft_neighb', 'sqft_habitable',  
                'view2', 'Cpl', 'Bmin', 'B', 'Bpl', 
                'zip006t011', 'zip014t024', 'zip040t053', 
                'zip070t077', 'zip107t115', 'zip116t122', 
                'zip146t168', 'thru2000', 
                'thru2020', 'thru60', 'thru80']

uppermedincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 
                  'lat', 'long', 'sqft_neighb', 'sqft_habitable', 
                  'D',  'Bmin', 'B', 'Bpl', 'Amin', 
                  'zip027t031', 'zip032t039', 'zip055t065',  
                  'zip125t144', 'zip146t168',  'thru2000', 
                  'thru2020', 'thru80', 'mi_5_scls']

highincome = ['bedrooms', 'bathrooms', 'waterfront', 'yr_renovated',
              'sqft_neighb','good',
              'Cmin','C', 'Cpl',  'B', 'Bpl', 'Amin', 
              'zip006t011', 'zip014t024', 'zip032t039', 'zip040t053', 
              'zip055t065', 'zip092t106', 'zip107t115', 'zip116t122', 
              'zip125t144', 'zip146t168', 'zip177t199', 'thru2000', 
              'thru2020', 'thru40', 'thru60', 'thru80', 
              'mi_nearest_scl', 'mi_5_scls']

def make_ols(df, x_columns, drops=None, target='pricesqrt', add_constant=False):
    if drops:
        drops.append(target)
        X = df.drop(columns=drops)
    else:
        X = df[x_columns]
    if add_constant:
        X = sm.add_constant(X)
    y = df[target]
    ols = sm.OLS(y, X)
    res = ols.fit()
    display(res.summary())
    fig = sm.graphics.qqplot(res.resid, dist=stats.norm, line='45', fit=True)
    return res


make_ols(lowtier,lowincome)
# make_ols(midtier,mediumincome)
# make_ols(uppermidtier,uppermedincome)
# make_ols(hightier,highincome)
X = lowtier[lowincome]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
list(zip(lowincome, vif))


print(type(hightier))
kc_data.columns
highincome


y = hightier[['pricesqrt']].copy()
X = hightier[['bedrooms', 'bathrooms', 'waterfront', 'yr_renovated',
              'sqft_neighb','good',
              'Cmin','C', 'Cpl',  'B', 'Bpl', 'Amin', 
              'zip006t011', 'zip014t024', 'zip032t039', 'zip040t053', 
              'zip055t065', 'zip092t106', 'zip107t115', 'zip116t122', 
              'zip125t144', 'zip146t168', 'zip177t199', 'thru2000', 
              'thru2020', 'thru40', 'thru60', 'thru80', 
              'mi_nearest_scl', 'mi_5_scls']].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)


print(len(X_train), len(X_test), len(y_train), len(y_test))


print(X_train)


linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)


train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test


mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
mse_test = np.sum((y_test-y_hat_test)**2)/len(y_test)
print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)


train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squarred Error:', train_mse)
print('Test Mean Squarred Error:', test_mse)


linreg.score(X_test, y_test)


y = m[['pricesqrt']]
X = kc_data.drop(['price', 'pricesqrt'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(len(X_train), len(X_test), len(y_train), len(y_test))


#print(X_train)


linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)


train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test


mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
mse_test = np.sum((y_test-y_hat_test)**2)/len(y_test)
print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)


train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squarred Error:', train_mse)
print('Test Mean Squarred Error:', test_mse)
print('Diff:', test_mse-train_mse)


linreg.score(X_test, y_test)


y = lowtier[['price']].copy()
X = lowtier[['bth', 'flr', 'lattrans', 'sqft_neighb', 'sqft_habitable','view3', 
             'Cmin', 'Cpl', 'Bmin', 'Bpl', 'zip032t039', 'zip055t065', 'zip070t077', 'zip092t106', 
             'zip107t115', 'zip116t122', 'zip125t144', 'zip146t168', 'thru2000', 
             'thru2020', 'thru40', 'thru60']].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)


print(len(X_train), len(X_test), len(y_train), len(y_test))


#print(X_train)


linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)


train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test


mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
mse_test = np.sum((y_test-y_hat_test)**2)/len(y_test)
print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)


train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squarred Error:', train_mse)
print('Test Mean Squarred Error:', test_mse)


print(sqrt(1031317247.0812349))
print(sqrt(1058736547.9719138))


linreg.score(X_test, y_test)


x_cols =['price', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'grade', 'yr_built', 'yr_renovated', 'zipcode', 'lat',
       'long', 'sqft_neighb', 'sqft_total', 'sqft_habitable',
       'con2', 'con3', 'con4', 'con5', 'view1', 'view2', 'view3', 'view4',
       'grd4', 'grd5', 'grd6', 'grd7', 'grd8', 'grd9', 'grd10', 'grd11',
       'grd12', 'grd13']


X = kc_data[highincome]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
list(zip(highincome, vif))


model.summary()


pd.set_option('display.max_rows', None)


kc_data = model.sort_values('coef', ascending=False)
kc_data.head(15)


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='grade', y='price', data=df_renovated)
ax.set(title='Grade relationship on Price', 
       xlabel='Grade', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='bathrooms', y='price', data=df_renovated)
ax.set(title='Bathrooms & Price', 
       xlabel='Bathrooms', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='bedrooms', y='price', data=df_renovated)
ax.set(title='Bedrooms & Price', 
       xlabel='Bedrooms', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='floors', y='price', data=df_renovated)
ax.set(title='Floors & Price', 
       xlabel='Floors', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='condition', y='price', data=df_renovated)
ax.set(title='Condition & Price', 
       xlabel='Condition', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.regplot(x='sqft_living', y='price', data=df_renovated)
ax.set(title='Square Feet Living Space & Price', 
       xlabel='SqFt.', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.regplot(x='sqft_above', y='price', data=df_renovated)
ax.set(title='Square Feet Above & Price', 
       xlabel='Sqft. Above', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='yr_built', y='price', data=df_renovated)
ax.set(title='Year Built & Price', 
       xlabel='Year Built', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='yr_renovated', y='price', data=df_renovated)
ax.set(title='Year Renovated & Price', 
       xlabel='Year', ylabel='Price')

fig.tight_layout()


df_renovated=data_pred.corr().abs().stack().reset_index().sort_values(0, ascending=False)
df_renovated['pairs'] = list(zip(df_renovated.level_0,df_renovated.level_1)
df_renovated.set_index(['pairs'], inplace = True)
df_renovated.drop(columns=['level_1','level_0'], inplace = True)
df_renovated.columns = ['cc']
df.drop_ducplicates(inplace=True)


df_renovated[(df.cc>.75) & (df.cc <1)]


mean_squared_error(y_train, predprice)



