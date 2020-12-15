import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic("matplotlib", " inline")
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import mlxtend
from scipy.stats import zscore
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
pd.options.display.max_rows = 4000
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor


# read data
kc_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 'sqft_above', 
              'sqft_basement', 'yr_built']

kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 
             'floors': float, 'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 
             'yr_built': int, 'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float}

kc_data = pd.read_csv('kc_house_data.csv', dtype = kc_dtypes, parse_dates = ['date'])

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
kc_data['waterfront'] = kc_data['waterfront'].fillna(146/19221)
kc_data['view'] = kc_data['view'].fillna(0)
kc_data['yr_renovated'] = kc_data['yr_renovated'].fillna(0)


#Convert to integer for whole number year
kc_data['yr_renovated'] = kc_data['yr_renovated'].astype('int')


#categorical variables
dumm = pd.get_dummies(kc_data['condition'], prefix='cond', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)
dumm = pd.get_dummies(kc_data['view'], prefix='view', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)
dumm = pd.get_dummies(kc_data['grade'], prefix='gra', drop_first=True, dtype=int)
kc_data = kc_data.merge(dumm, left_index=True, right_index=True)


#break up variables into diverse ranges
kc_data = kc_data.rename({'cond_2.0':'con2', 'cond_3.0':'con3','cond_4.0':'con4','cond_5.0':'con5'},axis=1)
kc_data = kc_data.rename({ 'view_1.0': 'view1', 'view_2.0': 'view2', 'view_3.0': 'view3', 'view_4.0':'view4'},axis=1)
kc_data = kc_data.rename({ 'gra_4': 'grd4', 'gra_5':'grd5', 'gra_6':'grd6',
       'gra_7':'grd7', 'gra_8':'grd8', 'gra_9':'grd9', 'gra_10':'grd10', 'gra_11':'grd11', 'gra_12':'grd12', 'gra_13':'grd13'},axis=1)


kc_data['zipcode'].sort_values()


kc_data['zip1'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]
kc_data['zip2'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]
kc_data['zip3'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]
kc_data['zip4'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]
kc_data['zip5'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]
kc_data['zip6'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]
kc_data['zip7'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]
kc_data['zip8'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]
kc_data['zip9'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]
kc_data['zip10'] = kc_data[(kc_data.zipcode > ) & (kc_data.zipcode<=)]


kc_data.info()


kc_data.isna().sum()


for col in kc_data.columns:
    try:
        print(col, kc_data[col].value_counts()[:5])
    except:
        print(col, kc_data[col].value_counts())
    print('\n')


kc_data = kc_data.drop('con3', axis=1).drop('grd13', axis=1)


kc_data.describe().round(3)


kc_data.hist(figsize=(10,10))
plt.tight_layout()


fig = pd.plotting.scatter_matrix(kc_data,figsize=(16,16));
print(type(fig))


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


# Initial Model - Price's Not Separated into Ranges
outcome = 'price'
x_cols = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'yr_built', 'yr_renovated', 'zipcode', 'lat',
       'long', 'sqft_total', 'sqft_habitable',
       'con2', 'con4', 'con5', 'view1', 'view2', 'view3', 'view4', 'grd4',
       'grd5', 'grd6', 'grd7', 'grd8', 'grd9', 'grd10', 'grd11', 'grd12']


predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=kc_data).fit()
model.summary()


model.params.sort_values()


# Adding in varaible income levels, each income has varaibles with cooresponding p values <0.05

lowtier = kc_data[kc_data.price <=300000]
midtier = kc_data[(kc_data.price > 300001) & (kc_data.price<=800000) ]
hightier = kc_data[kc_data.price >800000]

lowincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'zipcode', 'lat',
       'long', 'sqft_total', 'sqft_habitable', 'con4', 'con5', 'view1', 'grd4',
       'grd5', 'grd6']

mediumincome = ['bathrooms', 'waterfront', 'view', 'yr_built',
                  'zipcode', 'lat','long', 'sqft_habitable',
                   'con4', 'con5', 'view1','grd7', 'grd8', 'grd9', 
                  'grd10', 'grd11']
highincome = [ 'bathrooms', 'waterfront','yr_built', 
                   'yr_renovated', 'zipcode', 'lat','long', 'sqft_habitable',
                   'con4', 'con5', 'view1', 'grd4', 'grd6',
                   'grd7', 'grd8', 'grd9', 'grd10', 'grd11', 'grd12']

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
price_tiers = [
    ('low', lowtier, lowincome), 
    ('mid', midtier, mediumincome), 
    ('high', hightier, highincome)
]
for name, tier, income in price_tiers:
    print(name.upper())
    make_ols(tier, income)


# Attempt to find a more normalized price range
kc_columns = ['price']


for col in kc_columns:
    col_zscore = str(col + '_zscore')
    kc_data[col_zscore] = (kc_data[col] - kc_data[col].mean())/kc_data[col].std()
    kc_data = kc_data.loc[kc_data[col_zscore] < 2.25]
    kc_data = kc_data.loc[kc_data[col_zscore] > (-2.25)]
    kc_data = kc_data.drop(col_zscore, axis = 1)


plt.figure(figsize=(15,4))
plt.plot(kc_data['price'].value_counts().sort_index())


# Income Percentiles

for i in range(1,100):
    q = i / 100
    print('{} percentile: {}'.format(q, kc_data['price'].quantile(q=q)))


#Removed P values >= 0.05 relative to income tier

lowtier = kc_data[kc_data.price <=300000]
midtier = kc_data[(kc_data.price > 300001) & (kc_data.price<=800000) ]
hightier = kc_data[kc_data.price >800000]

lowincome = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'zipcode', 'lat',
       'long', 'sqft_total', 'sqft_habitable', 'con4', 'con5', 'view1', 'grd4',
       'grd5', 'grd6']

mediumincome = ['bathrooms', 'waterfront', 'view', 'yr_built',
                  'zipcode', 'lat','long', 'sqft_habitable',
                   'con4', 'con5', 'view1','grd7', 'grd8', 'grd9', 
                  'grd10', 'grd11']
highincome = [ 'bathrooms', 'waterfront', 'zipcode','long', 'sqft_habitable']

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
price_tiers = [
    ('low', lowtier, lowincome), 
    ('mid', midtier, mediumincome), 
    ('high', hightier, highincome)
]
for name, tier, income in price_tiers:
    print(name.upper())
    make_ols(tier, income)


kc_data.describe()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "bedrooms", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "bathrooms", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "floors", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "waterfront", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "view", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "yr_built", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "sqft_total", fig=fig)
plt.show()


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


# Function for Bedroom Visuals
def bedvisual(data,x,y):
    fig, ax = plt.subplots(figsize=(12,8))

    sns.boxplot(x=x, y=y, data=data)
    sns.regplot(x=x,y=y,data=data)
    ax.set(title=x.title()+' relationship on '+y.title(),
       xlabel=x.title(), ylabel='Price')

    fig.tight_layout()


# Function for Bathroom Visuals
def bathvisual(data,x,y):
    fig, ax = plt.subplots(figsize=(12,8))

    sns.boxplot(x=x, y=y, data=data)
    sns.regplot(x=x,y=y,data=data)
    ax.set(title=x.title()+' relationship on '+y.title(),
       xlabel=x.title(), ylabel='Price')

    fig.tight_layout()


bedvisual(lowtier,'bedrooms','price')


bedvisual(midtier,'bedrooms','price')


bedvisual(hightier,'bedrooms','price')


boxvisual(lowtier,'bedrooms','price')


bathvisual(lowtier,'bathrooms','price')


bathvisual(midtier,'bathrooms','price')


bathvisual(hightier,'bathrooms','price')


sns.catplot(x='waterfront', y='price', data=kc_data)
ax.set(title='Waterfront & Price', 
       xlabel='Waterfront', ylabel='Price')

fig.tight_layout()


sns.catplot(x='view', y='price', data=kc_data)
ax.set(title='View & Price', 
       xlabel='View', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.regplot(x='sqft_habitable', y='price', data=kc_data)
ax.set(title='Square Habitable & Price', 
       xlabel='SqFt.', ylabel='Price')

fig.tight_layout()


def coef(df, x_columns, drops=None, target='price', add_constant=False):
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
    return res


low = coef(kc_data,['bedrooms', 'bathrooms', 'floors', 'waterfront', 'zipcode', 'lat',
       'long', 'sqft_total', 'sqft_habitable', 'con4', 'con5', 'view1', 'grd4',
       'grd5', 'grd6'])


mid = coef(kc_data,['bathrooms', 'waterfront', 'view', 'yr_built',
                  'zipcode', 'lat','long', 'sqft_habitable',
                   'con4', 'con5', 'view1','grd7', 'grd8', 'grd9', 
                  'grd10', 'grd11'])


high = coef(kc_data, ['bathrooms', 'waterfront', 'zipcode','long', 'sqft_habitable'])


low.params.round(2).sort_values()


mid.params.round(2).sort_values()


high.params.round(2).sort_values()



