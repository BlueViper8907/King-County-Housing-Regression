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


def getClosest(home_lat: float, home_lon: float, dest_lat_series: 'series', dest_lon_series: 'series'):
    """Pass 1 set of coordinates and one latitude or longitude column you would like to compare it's distance to"""
    #radius of the earth in miles 
    r = 3963
    #setting variables to use to iterate through  
    closest = 100
    within_mile = 0
    i = 0
    #using a while loop to iterate over our data and calculate the distance between each datapoint and our homes 
    while i < dest_lat_series.size:
        lat_dist = radians(home_lat) - (dest_lat := radians(dest_lat_series.iloc[i]))
        lon_dist = radians(home_lon) - (radians(dest_lon_series.iloc[i]))
        a = sin(lat_dist / 2)**2 + cos(radians(home_lat)) * cos(radians(dest_lat)) * sin(lon_dist / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        c = r * c 
        #find the closest data to our homes by keeping our smallest (closest) value
        if (c < closest):
            closest = c
        #find all of the points that fall within one mile and count them 
        if (c <= 1.0):
            within_mile += 1
        i += 1
    return [closest, within_mile]


def plotcoef(model):
    """Takes in OLS results and returns a plot of the coefficients"""
    #make dataframe from summary of results 
    coef_df = pd.DataFrame(model.summary().tables[1].data)
    #rename your columns
    coef_df.columns = coef_df.iloc[0]
    #drop header row 
    coef_df = coef_df.drop(0)
    #set index to variables
    coef_df = coef_df.set_index(coef_df.columns[0])
    #change dtype from obj to float
    coef_df = coef_df.astype(float)
    #get errors
    err = coef_df['coef'] - coef_df['[0.025']
    #append err to end of dataframe 
    coef_df['errors'] = err
    #sort values for plotting 
    coef_df = coef_df.sort_values(by=['coef'])
    ## plotting time ##
    var = list(coef_df.index.values)
    #add variables column to dataframe 
    coef_df['var'] = var
    # define fig 
    fig, ax = plt.subplots(figsize=(8,5))
    #error bars for 95% confidence interval
    coef_df.plot(x='var', y='coef', kind='bar',
                ax=ax, fontsize=20, yerr='errors')
    #set title and label 
    plt.title('Coefficients of Features With 95% Confidence Interval', fontsize=30)
    ax.set_ylabel('Coefficients', fontsize=20)
    ax.set_xlabel(' ')
    #coefficients 
    ax.scatter(x= np.arange(coef_df.shape[0]),
              marker='o', s=80, 
              y=coef_df['coef'])
    return plt.show()


def make_ols(df, x_columns, target='price'):
    """Pass in a DataFrame & your predictive columns to return an OLS regression model """
    #set your x and y variables
    X = df[x_columns]
    y = df[target]
    # pass them into stats models OLS package
    ols = sm.OLS(y, X)
    #fit your model
    model = ols.fit()
    #display the model summarry
    display(model.summary())
    #plot the residuals 
    fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', alpha=.05, fit=True)
    #return model for later use 
    return model


#wrote up our data types to save on computer space and stop some of them from being inccorectly read as objs
kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 
             'floors': float, 'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 
             'yr_built': int, 'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float}


kc_data = pd.read_csv(r'~\Documents\Flatiron\data\data\kc_house_data.csv', parse_dates = ['date'], dtype=kc_dtypes)
schools = pd.read_csv(r'~\Documents\Flatiron\data\data\Schools.csv')
foods = pd.read_csv(r'~\Documents\Flatiron\foods.csv')


foods = foods.loc[foods['lat'] get_ipython().getoutput("= '[0.0]'].copy()")
foods = foods.loc[foods['long'] get_ipython().getoutput("= '[0.0]'].copy()")
foods['lat'] = foods['lat'].astype(dtype=float)
foods['long'] = foods['long'].astype(dtype=float)


rest = foods.loc[foods['SEAT_CAP'] get_ipython().getoutput("= 'Grocery']")
groc = foods.loc[foods['SEAT_CAP'] == 'Grocery']


kc_dict = {}


i = 0
while i < kc_data['lat'].size:
    school = getClosest(kc_data['lat'].iloc[i], kc_data['long'].iloc[i], schools['LAT_CEN'], schools['LONG_CEN'])
    restaurant = getClosest(kc_data['lat'].iloc[i], kc_data['long'].iloc[i], rest['lat'], rest['long'])
    grocery = getClosest(kc_data['lat'].iloc[i], kc_data['long'].iloc[i], groc['lat'], groc['long'])
    kc_dict[i] = {
        "closest school": school[0],
        "schools within mile": school[1],
        "closest restaurant": restaurant[0],
        "restaurants within mile": restaurant[1],
        "closest grocery": grocery[0],
        "groceries within mile": grocery[1]}
    i += 1 


kc = pd.DataFrame.from_dict(kc_dict, orient='index')
kc_data = kc_data.merge(kc, left_index=True, right_index=True)


kc_data = kc_data.rename(columns ={'closest school': 'mi_2_scl', 'schools within mile': 'scls_in_mi', 'closest restaurant':'mi_2_rest', 
                          'restaurants within mile':'rest_in_mi','closest grocery': 'mi_2_groc', 'groceries within mile': 'groc_in_mi'})


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


hist = kc_data[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 
                'lat', 'long', 'sqft_living15', 'sqft_lot15', 'mi_2_scl', 'scls_in_mi', 'mi_2_rest',
                'rest_in_mi', 'mi_2_groc', 'groc_in_mi']]
hist.hist(figsize=(15,15))
plt.tight_layout()


# fig = pd.plotting.scatter_matrix(kc_data,figsize=(16,16));
kc_data.columns


fig, ax = plt.subplots(figsize=(25,20))
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
all_data = kc_data.copy()
kc_data = kc_data[['price', 'bedrooms', 'bathrooms', 'floors','waterfront', 
                   'yr_renovated', 'lat', 'long', 
                   'sqft_total', 'sqft_neighb', 'sqft_habitable', 
                   'good', 'view1', 'view2', 'view3', 'view4', 
                   'D', 'Cmin', 'C', 'Cpl', 'Bmin', 'B', 'Bpl', 'Amin', 
                   'zip006t011', 'zip014t024', 'zip027t031', 'zip032t039', 
                   'zip040t053', 'zip055t065', 'zip070t077', 'zip092t106', 
                   'zip107t115', 'zip116t122', 'zip125t144', 'zip146t168', 
                   'zip177t199', 
                   'thru2000', 'thru2020', 'thru40', 'thru60', 'thru80',
                   'mi_2_scl', 'scls_in_mi', 'mi_2_rest', 'rest_in_mi', 'mi_2_groc', 'groc_in_mi']].copy()


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
          'mi_2_scl', 'scls_in_mi', 'mi_2_rest', 'rest_in_mi', 'mi_2_groc', 'groc_in_mi']

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
          'mi_2_scl', 'scls_in_mi', 'mi_2_rest', 'rest_in_mi', 'mi_2_groc', 'groc_in_mi']

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
          'mi_2_scl', 'scls_in_mi', 'mi_2_rest', 'rest_in_mi', 'mi_2_groc', 'groc_in_mi']


price_tiers = [('low', lowtier, lowincome), 
               ('mid', midtier, mediumincome), 
               ('high', hightier, highincome)]


for name, tier, income in price_tiers:
    print(name.upper())
    make_ols(tier, income)


for col in ['price']:
    col_zscore = str(col + '_zscore')
    kc_data[col_zscore] = (kc_data[col] - kc_data[col].mean())/kc_data[col].std()
    kc_data = kc_data.loc[kc_data[col_zscore] < 2]
    kc_data = kc_data.loc[kc_data[col_zscore] > (-2)]
    kc_data = kc_data.drop(col_zscore, axis = 1)


plt.figure(figsize=(15,4))
plt.plot(kc_data['price'].value_counts().sort_index())


for i in range(1,100):
    q = i / 100
    print('{} percentile: {}'.format(q, kc_data['price'].quantile(q=q)))


#in bedrooms, we can clearly see a single outlier that is likely just a typo 
kc_data[kc_data['bedrooms'] == 33]
# wouldn't be realistic for a house with 33 bedrooms to only have a sqft_living of 1620 and only 1 3/4 bathrooms so we will adjust to 3 
kc_data[kc_data['bedrooms'] == 33] = kc_data[kc_data['bedrooms'] == 33].replace(33,3)


# to fix other outliers we will explore our data and find cutoffs that seem reasonable 
kc_data = kc_data.loc[kc_data['sqft_total'] <= 1.000000e+09] 
kc_data = kc_data.loc[kc_data['sqft_total'] >= 400000]
kc_data = kc_data.loc[kc_data['sqft_neighb'] <= 1.000000e+09]
kc_data = kc_data.loc[kc_data['sqft_habitable'] >= 400000]
kc_data = kc_data.loc[kc_data['sqft_habitable'] <= 1.000000e+07]
kc_data =  kc_data.loc[kc_data['bathrooms'] >= 1]
kc_data =  kc_data.loc[kc_data['bathrooms'] <= 5]
kc_data =  kc_data.loc[kc_data['bedrooms'] <= 7]
kc_data.columns


lowtier = kc_data[(kc_data.price >= 210000) & (kc_data.price <= 348000) ]
midtier = kc_data[(kc_data.price >= 348000) & (kc_data.price <= 480000) ]
uppermidtier = kc_data[(kc_data.price >= 480000) & (kc_data.price <= 640000) ]
hightier = kc_data[(kc_data.price >= 640000) & (kc_data.price <= 900000)]


lowincome = ['bathrooms', 'waterfront', 'lat', 'long',
             'sqft_total', 'sqft_habitable', 
             'view1', 'view2', 'view3', 
             'C', 'Cpl', 'Bmin', 'B',
             'zip040t053', 'zip055t065', 'zip092t106', 
             'zip107t115', 'zip146t168', 
             'groc_in_mi']

mediumincome = ['bathrooms',  'lat', 'long', 
                'sqft_habitable', 'view2',   
                'Cpl', 'Bmin', 'B', 'Bpl',   
                'zip006t011', 'zip014t024', 'zip032t039', 
                'zip055t065', 'zip070t077', 'zip092t106', 
                'zip177t199', 'rest_in_mi', 'groc_in_mi',
                'thru2000', 'thru2020', 'thru60', 'thru80']

uppermedincome = ['bathrooms',  'lat', 'sqft_habitable',   
                  'C', 'Bmin', 'B', 
                  'zip014t024', 'zip027t031', 'zip032t039', 
                  'zip070t077', 'zip125t144', 'zip146t168', 
                  'thru2000', 'thru2020', 'thru60', 'thru80']


highincome = ['bathrooms', 'floors', 'sqft_neighb', 
              'sqft_habitable', 'thru2020',
              'zip006t011', 'zip107t115',
              'zip116t122', 'zip177t199', 
              'mi_2_scl', 'scls_in_mi', 'mi_2_rest',
              'mi_2_groc', 'groc_in_mi']


price_tiers = [('low', lowtier, lowincome), 
               ('mid', midtier, mediumincome), 
               ('upmid', uppermidtier, uppermedincome),
               ('high', hightier, highincome)]


for name, tier, income in price_tiers:
    print(name.upper())
    make_ols(tier, income)


#first step
high_data = hightier[['price', 'bathrooms', 'floors', 'sqft_neighb', 
                      'sqft_habitable', 'thru2020',
                      'zip006t011', 'zip107t115',
                      'zip116t122', 'zip177t199', 
                      'mi_2_scl', 'scls_in_mi', 'mi_2_rest',
                      'mi_2_groc', 'groc_in_mi']].copy()

training_data, testing_data = train_test_split(high_data, test_size=0.25, random_state=44)


#split columns
target = 'price'
predictive_cols = training_data.drop('price', 1).columns


high_model = make_ols(hightier, predictive_cols)


# predictions
y_pred_train = high_model.predict(training_data[predictive_cols])
y_pred_test = high_model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)
plotcoef(high_model)


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(high_model, "groc_in_mi", fig=fig)
plt.show()


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(high_model, "sqft_habitable", fig=fig)
plt.show()


#first step
upper_med_data = uppermidtier[['bathrooms',  'lat', 'sqft_habitable',   
                               'C', 'Bmin', 'B', 'price',
                               'zip014t024', 'zip027t031', 'zip032t039', 
                               'zip070t077', 'zip125t144', 'zip146t168', 
                               'thru2000', 'thru2020', 'thru60', 'thru80']].copy()

training_data, testing_data = train_test_split(upper_med_data,test_size=0.30, random_state=55)


#split columns
target = 'price'
predictive_cols = training_data.drop('price', 1).columns


uppmid_model = make_ols(training_data, predictive_cols)


# predictions
y_pred_train = uppmid_model.predict(training_data[predictive_cols])
y_pred_test = uppmid_model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)
plotcoef(uppmid_model)


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(uppmid_model, "sqft_habitable", fig=fig)
plt.show()


#first step
mid_data = midtier[['bathrooms',  'lat', 'long', 
                    'sqft_habitable', 'view2', 'price', 
                    'Cpl', 'Bmin', 'B', 'Bpl',   
                    'zip006t011', 'zip014t024', 'zip032t039', 
                    'zip055t065', 'zip070t077', 'zip092t106', 
                    'zip177t199', 'rest_in_mi', 'groc_in_mi',
                    'thru2000', 'thru2020', 'thru60', 'thru80']].copy()
training_data, testing_data = train_test_split(mid_data, test_size=0.30, random_state=70)


#split columns
target = 'price'
predictive_cols = training_data.drop('price', 1).columns


mid_model = make_ols(mid_data, predictive_cols)


# predictions
y_pred_train = mid_model.predict(training_data[predictive_cols])
y_pred_test = mid_model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)
plotcoef(mid_model)


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(high_model, "sqft_habitable", fig=fig)
plt.show()


#first step
low_data = lowtier[['bathrooms', 'waterfront', 'lat', 'long',
                    'sqft_total', 'sqft_habitable', 
                    'view1', 'view2', 'view3', 
                    'C', 'Cpl', 'Bmin', 'B', 'price',
                    'zip040t053', 'zip055t065', 'zip092t106', 
                    'zip107t115', 'zip146t168', 
                    'groc_in_mi']].copy()

training_data, testing_data = train_test_split(low_data, test_size=0.25, random_state=66)


#split columns
target = 'price'
predictive_cols = training_data.drop('price', 1).columns


model = make_ols(low_data, predictive_cols)


# predictions
y_pred_train = model.predict(training_data[predictive_cols])
y_pred_test = model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)
plotcoef(low_model)


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(high_model, "sqft_habitable", fig=fig)
plt.show()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='mi_2_scl', y='price', data=kc_data)
ax.set(title='Miles To Nearest School & Relationship to Price', 
       xlabel='Miles', ylabel='Price')

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

sns.regplot(x='sqft_habitable', y='price', data=kc_data)
ax.set(title='Square Habitable & Price', 
       xlabel='SqFt.', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='yr_built', y='price', data=all_data)
ax.set(title='Year Built & Price', 
       xlabel='Year Built', ylabel='Price')

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='yr_renovated', y='price', data=all_data)
ax.set(title='Year Renovated & Price', 
       xlabel='Year', ylabel='Price')

fig.tight_layout()






