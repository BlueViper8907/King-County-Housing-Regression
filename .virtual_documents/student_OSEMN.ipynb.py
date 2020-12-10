import pandas as pd
import numpy as np
import seaborn as sns
import mlxtend


import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from math import sin, cos, sqrt, atan2, radians
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn import svm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor


#set settings 
get_ipython().run_line_magic("matplotlib", " inline")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# read data
kc_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 'sqft_above', 
              'sqft_basement', 'yr_built']

kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 
             'floors': float, 'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 
             'yr_built': int, 'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float}

kc_data = pd.read_csv(r'~\Documents\Flatiron\data\data\kc_house_data.csv', parse_dates = ['date'], dtype=kc_dtypes)


kc_data['sqft_basement'] = kc_data['sqft_basement'].replace({'?': 0})
kc_data['sqft_basement'] = kc_data['sqft_basement'].astype(dtype=float, errors='ignore')
kc_data['sqft_total'] = kc_data['sqft_living']*kc_data['sqft_lot']
kc_data['sqft_habitable'] = (kc_data['sqft_above']+1)*(kc_data['sqft_basement']+1)


#drop unnessecary columns and fix data
kc_data = kc_data.drop('sqft_living', 1).drop('sqft_lot', 1).drop('id', 1).drop('date', 1).drop('sqft_above',1).drop('sqft_basement',1)

#look for outliers, in bedrooms, we can clearly see a single outlier, for other columns, filtering by z score will be easiest 
kc_data[kc_data['bedrooms'] == 33]
# wouldn't be realistic for a house with 33 bedrooms to only have a sqft_living of 1620 and only 1 3/4 bathrooms so it looks like a typo
# will adjust to 3 
kc_data[kc_data['bedrooms'] == 33] = kc_data[kc_data['bedrooms'] == 33].replace(33,3)


#setting waterfront NaN values equal to the ratio of waterfront/non-waterfront properties, will want to try and narrow by zipcode 
#filling NaN with easily seperatable/changable values helpful 
kc_data['waterfront'] = kc_data['waterfront'].fillna(0)
kc_data['view'] = kc_data['view'].fillna(0)
kc_data['yr_renovated'] = kc_data['yr_renovated'].fillna(0)


#Convert to integer for whole number year
kc_data['yr_renovated'] = kc_data['yr_renovated'].astype('int')


dumm = pd.get_dummies(kc_data['condition'], prefix='cond', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)
dumm = pd.get_dummies(kc_data['view'], prefix='view', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)
dumm = pd.get_dummies(kc_data['grade'], prefix='gra', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)


kc_data = kc_data.rename({'cond_2.0':'con2', 'cond_3.0':'con3','cond_4.0':'con4','cond_5.0':'con5'},axis=1)
kc_data = kc_data.rename({ 'view_1.0': 'view1', 'view_2.0': 'view2', 'view_3.0': 'view3', 'view_4.0':'view4'},axis=1)
kc_data = kc_data.rename({ 'gra_4': 'grd4', 'gra_5':'grd5', 'gra_6':'grd6',
       'gra_7':'grd7', 'gra_8':'grd8', 'gra_9':'grd9', 'gra_10':'grd10', 'gra_11':'grd11', 'gra_12':'grd12', 'gra_13':'grd13'},axis=1)


zips = []

for zipcode in kc_data.zipcode:
    zips.append(zipcode)
    
zips = list(set(zips))
zips.sort()


#would be a bad idea to add 70 dummies to our columns so instead we're binning our zipcodes by 5 
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:5], value= 'zip1')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:10], value= 'zip2')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:15], value= 'zip3')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:20], value= 'zip4')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:25], value= 'zip5')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:30], value= 'zip6')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:35], value= 'zip7')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:40], value= 'zip8')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:45], value= 'zip9')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:50], value= 'zip10')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:55], value= 'zip11')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:60], value= 'zip12')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:65], value= 'zip13')
kc_data['zipcode'] = kc_data.zipcode.replace(to_replace = zips[:70], value= 'zip14')


years = []

for year in kc_data.yr_built:
    years.append(year)
    
years = list(set(years))
years.sort()


#gonna do the same for year built by 20 years, will give us 6 new columns, may be illuminating 
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[:20], value= 'thru20')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[:40], value= 'thru40')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[:60], value= 'thru60')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[:80], value= 'thru80')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[:100], value= 'thru2000')
kc_data['yr_built'] = kc_data.yr_built.replace(to_replace = years[:120], value= 'thru2020')


dumm = pd.get_dummies(kc_data['zipcode'], prefix=None, drop_first=True)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)
dumm = pd.get_dummies(kc_data['yr_built'], prefix=None, drop_first=True)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)


kc_data =  kc_data.loc[kc_data['bathrooms'] >= 1]
kc_data.columns


kc_data = kc_data[['price', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'yr_renovated', 
                   'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_total', 'sqft_habitable', 'con2', 
                   'con3', 'con4', 'con5', 'view1', 'view2', 'view3', 'view4', 'grd4', 'grd5', 'grd6', 
                   'grd7', 'grd8', 'grd9', 'grd10', 'grd11', 'grd12', 'grd13', 'zip10', 'zip11', 'zip12', 
                   'zip13', 'zip14', 'zip2', 'zip3', 'zip4', 'zip5', 'zip6', 'zip7', 'zip8', 'zip9',
                   'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80']].copy()


schools = pd.read_csv(r'~\Documents\Flatiron\data\data\Schools.csv')
schools.info()


type(schools['LAT_CEN'].iloc[2])


#calculate distance between schools and data 
kc = {}
kc3 = {}
kc5 = {}
# approximate radius of earth in miles  miles
i = 0
while i <= 21521:
    R = 3963.0
    k = 0
    lat1 = radians(kc_data['lat'].iloc[i])
    lon1 = radians(kc_data['long'].iloc[i])
    distance = []
    
    while k <= 641:
        lat2 = radians(schools['LAT_CEN'].iloc[k])
        lon2 = radians(schools['LONG_CEN'].iloc[k])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance.append(R * c)
        
        k += 1 
        
    distance.sort()
    kc[i] = distance[0:1]
    kc3[i] = sum(distance[0:3])
    kc5[i] = sum(distance[0:5])
    i += 1


kc1 = pd.DataFrame.from_dict(kc, orient='index', columns=['mi_nearest_scl'])
kc3 = pd.DataFrame.from_dict(kc3, orient='index', columns=['mi_3_scls'])
kc5 = pd.DataFrame.from_dict(kc5, orient='index', columns=['mi_5_scls'])


kc_data = kc_data.merge(kc1, left_index=True, right_index=True)
kc_data = kc_data.merge(kc3, left_index=True, right_index=True)
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


# fig = pd.plotting.scatter_matrix(kc_data,figsize=(16,16));
# print(type(fig))


fig, ax = plt.subplots(figsize=(26,12))

corr = kc_data.corr().abs().round(3)

mask = np.triu(np.ones_like(corr, dtype=np.bool))

sns.heatmap(corr, annot=True, mask=mask, cmap='Oranges', ax=ax)
plt.setp(ax.get_xticklabels(), 
         rotation=45, 
         ha="right",
         rotation_mode="anchor")
ax.set_title('Correlations')


outcome = 'price'
x_cols = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'yr_renovated', 
                   'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_total', 'sqft_habitable', 'con2', 
                   'con4', 'con5', 'view1', 'view2', 'view3', 'view4', 'grd4', 'grd5', 'grd6', 
                   'grd7', 'grd8', 'grd9', 'grd10', 'grd11', 'grd12', 'zip10', 'zip11', 'zip12', 
                   'zip13', 'zip14', 'zip2', 'zip3', 'zip4', 'zip5', 'zip6', 'zip7', 'zip8', 'zip9',
                   'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80', 'mi_3_scls']


predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=kc_data).fit()
model.summary()


model.params.sort_values()


lowtier = kc_data[kc_data.price <=300000]
midtier = kc_data[(kc_data.price > 300001) & (kc_data.price<=800000) ]
hightier = kc_data[kc_data.price >800000]

lowincome = ['bedrooms', 'mi_nearest_scl',	'mi_3_scls', 'mi_5_scls',
                   'lat', 'long', 'sqft_living15', 'sqft_total', 'sqft_habitable', 'con2', 
                   'con4', 'con5', 'view1', 'view2', 'grd4', 'grd10', 'grd11', 'grd12', 'zip10', 'zip11', 'zip12', 
                   'zip13', 'zip14', 'zip2', 'zip3', 'zip4', 'zip5', 'zip6', 'zip7', 'zip8', 'zip9',
                   'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80']

mediumincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'yr_renovated', 
                   'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_total', 'sqft_habitable', 'con2', 
                   'con4', 'con5', 'view1', 'view2', 'view3', 'view4', 'grd4', 'grd5', 'grd6', 
                   'grd7', 'grd8', 'grd9', 'grd10', 'grd11', 'grd12', 'zip10', 'zip11', 'zip12', 
                   'zip13', 'zip14', 'zip2', 'zip3', 'zip4', 'zip5', 'zip6', 'zip7', 'zip8', 'zip9',
                   'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80','mi_nearest_scl',	
                   'mi_3_scls', 'mi_5_scls']

highincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'yr_renovated', 
                   'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_total', 'sqft_habitable', 'con2', 
                   'con4', 'con5', 'view1', 'view2', 'view3', 'view4', 'grd4', 'grd5', 'grd6', 
                   'grd7', 'grd8', 'grd9', 'grd10', 'grd11', 'grd12', 'zip10', 'zip11', 'zip12', 
                   'zip13', 'zip14', 'zip2', 'zip3', 'zip4', 'zip5', 'zip6', 'zip7', 'zip8', 'zip9',
                   'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80', 'mi_nearest_scl',	'mi_3_scls', 'mi_5_scls']

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
    res = ols.fit()
    display(res.summary())
    fig = sm.graphics.qqplot(res.resid, dist=stats.norm, line='45', fit=True)
    return res
make_ols(lowtier,lowincome)
make_ols(midtier,mediumincome)
make_ols(hightier,highincome)


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


lowtier = kc_data[(kc_data.price > 210000) & (kc_data.price<=348000) ]
midtier = kc_data[(kc_data.price > 348000) & (kc_data.price<=480000) ]
uppermidtier = kc_data[(kc_data.price > 480000) & (kc_data.price<=664000) ]
hightier = kc_data[(kc_data.price >664000) & (kc_data.price<=1080000)]

lowincome = ['bedrooms', 'bathrooms', 'floors', 'lat', 'long', 'sqft_lot15', 'sqft_total', 
             'sqft_habitable',  'view1', 'view2', 'view3',  'grd7', 'grd8', 'grd9', 'zip12', 
             'zip13', 'zip5',  'zip7', 'zip8', 'zip9', 'thru80']


mediumincome = ['bathrooms', 'lat', 'long', 'sqft_living15',  'sqft_total', 'sqft_habitable', 
              'view2', 'view3', 'grd9', 'grd10', 'zip10', 'zip11', 'zip2', 'zip3', 'zip6', 'zip8',
              'thru2000', 'thru2020', 'thru60', 'thru80']

uppermedincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'lat', 'long', 'sqft_living15',
                  'sqft_total', 'sqft_habitable', 'zip12', 'zip13', 'zip2', 'zip3', 'zip4', 'zip5',
                  'zip6', 'zip7', 'zip8', 'thru2000', 'thru2020', 'thru60', 'thru80']

highincome = ['bathrooms', 'bathrooms', 'floors', 'waterfront', 'yr_renovated', 'lat', 'long', 
              'sqft_living15','sqft_habitable', 'view2', 'grd4','zip11', 'zip12', 'zip13', 
              'zip14', 'zip2', 'zip3', 'zip4', 'zip5', 'zip6', 'zip7', 'zip8']

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
    res = ols.fit()
    display(res.summary())
    fig = sm.graphics.qqplot(res.resid, dist=stats.norm, line='45', fit=True)
    return res


#make_ols(lowtier,lowincome)
#make_ols(midtier,mediumincome)
#make_ols(uppermidtier,uppermedincome)
make_ols(hightier,highincome)


print(type(hightier))
kc_data.columns
highincome


y = hightier[['price']].copy()
X = hightier[['bedrooms', 'bathrooms', 'floors', 'waterfront',
              'lat', 'long', 'sqft_living15', 'sqft_total', 'sqft_habitable',
              'con4', 'view2', 'view3', 'view4', 
              'grd6', 'grd8', 'grd9', 'grd10', 'grd11', 
              'zip10', 'zip11', 'zip12', 'zip13', 'zip14', 'zip2', 'zip3', 'zip4', 'zip5', 
              'zip6', 'zip7', 'zip8', 'zip9', 'thru2000', 'thru2020',  
              'thru80', 'mi_5_scls']].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


print(len(X_train), len(X_test), len(y_train), len(y_test))


print(X_train)


reg = LinearRegression()
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


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)


y = midtier[['price']]
X = midtier.drop(['price'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


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
print('Diff:', test_mse-train_mse)



linreg.score(X_test, y_test)


y = lowtier[['price']]
X = lowtier.drop(['price'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


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


x_cols =['price', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'grade', 'yr_built', 'yr_renovated', 'zipcode', 'lat',
       'long', 'sqft_living15', 'sqft_lot15', 'sqft_total', 'sqft_habitable',
       'con2', 'con3', 'con4', 'con5', 'view1', 'view2', 'view3', 'view4',
       'grd4', 'grd5', 'grd6', 'grd7', 'grd8', 'grd9', 'grd10', 'grd11',
       'grd12', 'grd13']


X = kc_data[x_cols]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
list(zip(x_cols, vif))


model.summary()


pd.set_option('display.max_rows', None)


kc_data = kc_data.sort_values('coef', ascending=False)
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


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


X = kc_data.drop('price', axis=1)
y = kc_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ols1 = LinearRegression()
ols1.fit(X_train, y_train)

predprice = ols1.predict(y_train)

np.sqrt(mean_squared_error(X_train, predprice))

ols2 = LinearRegression()
ols_cv_mse = cross_val_score(ols2, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
ols_cv_mse.mean()



mean_squared_error(y_train, predprice)



