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
from scipy.stats import zscore
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# assigning columns we'll be interpreting later
kc_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 'sqft_above', 
              'sqft_basement', 'yr_built']
#assign data types so pandas can read everything properly and to save your computer a bit of time 
kc_dtypes = {'id': int, 'date' : float,  'price': int, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 
             'floors': float, 'waterfront': float, 'view' : float, 'condition': int, 'grade': int, 'sqft_above': int, 
             'yr_built': int, 'yr_renovated': int, 'zipcode': int, 'lat': float, 'long': float}
#read your data, parse your dates and read the data types 
kc_data = pd.read_csv(r'~\Documents\Flatiron\data\data\Cleaned_Dataset.csv', parse_dates = ['date'], dtype=kc_dtypes)


#after finding out where our data was sourced from, we can confirm with significant certainty that the null values
#should have been 0, replacing these here and changing it to an int 
kc_data['waterfront'] = kc_data['waterfront'].replace(to_replace='0.007596', value='0')
kc_data['waterfront'] = kc_data['waterfront'].astype(int)





lowtier = kc_data[kc_data.price <=300000]
midtier = kc_data[(kc_data.price > 300001) & (kc_data.price<=800000) ]
hightier = kc_data[kc_data.price >800000]


# kc_data.info()
lowtier.info()


kc_data.isna().sum()


#Drop NaN
kc_data = kc_data.copy()
df_renovated = kc_data
df_renovated


#Ensuring no strings in yr_renovated
#df_renovated['yr_renovated'] = df_renovated['yr_renovated'].astype('float')
#Convert to integer for whole number year
#df_renovated['yr_renovated'] = df_renovated['yr_renovated'].astype('int')
#Convert Zipcode to Float
#df_renovated['zipcode'] = df_renovated['zipcode'].astype('float')
#Convert Condition to float
#df_renovated['condition'] = df_renovated['condition'].astype('float')


df_renovated.info()


#df_renovated['date'] = pd.to_datetime(df_renovated['date'])
#df_renovated['date']


for col in df_renovated.columns:
    try:
        print(col, df_renovated[col].value_counts()[:5])
    except:
        print(col, df_renovated[col].value_counts())
    print('\n')


duplicates = df_renovated[df_renovated.duplicated()]
print(len(duplicates))
duplicates.head()


# Value counts for variables
for col in df_renovated.columns:
    print(col, '\n', df_renovated[col].value_counts(normalize=True).head(), '\n\n')


def check_column(df_renovated, col_name, n_unique=10):
    print('Datatype')
    print('\t',df_renovated[col_name].dtypes)
    
    num_nulls = df_renovated[col_name].isna().sum()
    print(f'Null Values Present = {num_nulls}')
    
    display(df_renovated[col_name].describe().round(3))
    
    print('\nValue Counts:')
    display(df_renovated[col_name].value_counts(n_unique))
    
check_column(df_renovated,'price')


df_renovated.describe().round(3)


df_renovated.hist(figsize=(10,10))
plt.tight_layout()


#fig = pd.plotting.scatter_matrix(df_renovated,figsize=(16,16));
#print(type(fig))


fig, ax = plt.subplots(figsize=(16,12))

corr = df_renovated.corr().abs().round(3)

mask = np.triu(np.ones_like(corr, dtype=np.bool))

sns.heatmap(corr, annot=True, mask=mask, cmap='Oranges', ax=ax)
plt.setp(ax.get_xticklabels(), 
         rotation=45, 
         ha="right",
         rotation_mode="anchor")
ax.set_title('Correlations')


cols_to_plot = ["bedrooms", "bathrooms","grade","floors", "condition", "sqft_living"]

colors = ['red', 'orange', 'magenta', 'green', 'blue','red']
fig, axes = plt.subplots(ncols=6, figsize=(20,6))

for i, col in enumerate(cols_to_plot):
    axes[i].scatter(x=df_renovated[col],y=df_renovated['price'],c=colors[i],marker='.')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('price')
    
plt.tight_layout()


sns.regplot(x= 'grade', y= 'price', data=df_renovated)


sns.regplot(x= 'bedrooms', y= 'price', data=df_renovated)


sns.regplot(x= 'bathrooms', y= 'price', data=df_renovated)


sns.catplot(x= 'condition', y= 'price', data=df_renovated)


sns.catplot(x= 'floors', y= 'price', data=df_renovated)


sns.regplot(x= 'sqft_living', y= 'price', data=df_renovated)


sns.regplot(x= 'sqft_lot', y= 'price', data=df_renovated)


sns.catplot(x= 'waterfront', y= 'price', data=df_renovated)


sns.catplot(x= 'view', y= 'price', data=df_renovated)


sns.regplot(x= 'sqft_above', y= 'price', data=df_renovated)


sns.regplot(x= 'sqft_basement', y= 'price', data=df_renovated)


sns.regplot(x= 'lat', y= 'price', data=df_renovated)


sns.regplot(x= 'long', y= 'price', data=df_renovated)


# df_renovated.info()
test = test['id'].drop_duplicates()


# read data
kc_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 'sqft_above', 
              'sqft_basement', 'yr_built']

kc_dtypes = {'id': int, 'date' : float,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 
             'floors': float, 'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 
             'yr_built': int, 'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float}

kc_data = pd.read_csv('Notebooks/data/Cleaned_Dataset.csv', dtype = kc_dtypes, parse_dates = ['date'])
# kc_data[kc_data['bedrooms'] == 33] = kc_data[kc_data['bedrooms'] == 33].replace(33,3)
# kc_data['sqft_basement'] = kc_data['sqft_basement'].replace({'?': 0})
# kc_data['sqft_basement'] = kc_data['sqft_basement'].astype(dtype=float, errors='ignore')
# kc_data = kc_data['id'].drop_duplicates()


outcome = 'price'
x_cols = ['bedrooms','bathrooms','sqft','view']


test = midtier.copy()
# test['sqft'] = test['sqft_above']* test['sqft_living']
# test = test[(test.bathrooms >=1)]
# test = test[(test.bedrooms <= 6)]
# test = test[(test.sqft_lot >= 1000)]
# test = test[(test.sqft_lot <= 10000)]
# df =  pd.get_dummies(test['condition'], drop_first = True,prefix = 'con_dummy')
# for col in kc_columns:
#     col_zscore = str(col + '_zscore')
#     kc_data[col_zscore] = (kc_data[col] - kc_data[col].mean())/kc_data[col].std()
#     kc_data = kc_data.loc[kc_data[col_zscore] < 2]
#     kc_data = kc_data.loc[kc_data[col_zscore] > (-2)]
#     kc_data = kc_data.drop(col_zscore, axis = 1)

# test.describe()
# test = test.merge(df, left_index = True,right_index = True)


# predictors = '+'.join(x_cols)
# formula = outcome + '~' + predictors
# model = sm.OLS(formula,lowtier).fit()
# model.summary()

def make_ols(df, x_columns, drops=None, target='price', add_constant=True):
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

make_ols(hightier,[ 'bedrooms', 'bathrooms', 'sqft_living', 'condition', 'grade', 'sqft_above'])


fig = sm.graphics.qqplot(res.resid, dist=stats.norm, line='45', fit=True)


model.params.sort_values()


plt.figure(figsize=(15,4))
plt.plot(df_renovated['price'].value_counts().sort_index())


for i in range(90,100):
    q = i / 100
    print('{} percentile: {}'.format(q, df_renovated['price'].quantile(q=q)))


#plt.scatter(model.predict([x_cols]), model.resid)
#plt.plot(model.predict([x_cols]), [0 for i in range(len(subset))])


fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)


y = df_renovated[['price']]
X = df_renovated.drop(['price'], axis=1)


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


z = np.abs(stats.zscore(df_renovated["price"]))
print(np.where(z > 3))


model.summary()


pd.set_option('display.max_rows', None)


df_renovated = df_renovated.sort_values('coef', ascending=False)
df_renovated.head(15)


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
