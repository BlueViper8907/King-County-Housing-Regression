import pandas as pd
import numpy as np
import seaborn as sns
import mlxtend


import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.patches as mpatches


from math import sin, cos, sqrt, atan2, radians
from sklearn import svm
from cycler import cycler
from matplotlib import rcParams
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


# for the sake of reading this notebook, we'll be removing these settings before running through, 
# but when you're exploring your own data or checking out our work here,
# we'd reccomend increasing all of your output displays so you can really see all your data
# pd.set_option('display.max_rows', 50)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


get_ipython().run_line_magic("matplotlib", " inline")
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.color'] = '#FBE122'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.color'] = '#A2AAAD'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['axes.labelcolor'] = '#A2AAAD'
mpl.rcParams['axes.facecolor'] = 'black'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#2C5234", "#46664d", "#687a6c"]) 
mpl.rcParams['figure.facecolor'] = 'black'
mpl.rcParams['figure.edgecolor'] = 'black'   
mpl.rcParams['image.cmap'] = 'YlGn'
mpl.rcParams['text.color'] = '#A2AAAD'
mpl.rcParams['xtick.color'] = '#A2AAAD'
mpl.rcParams['ytick.color'] = '#A2AAAD'


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


# this function is directly from the matplotlib documentation 
# https://matplotlib.org/_modules/matplotlib/figure.html#Figure.set_frameon
def set_frameon(self, b):
        """
        Set the figure's background patch visibility, i.e.
        whether the figure background will be drawn. Equivalent to
        ``Figure.patch.set_visible()``.

        Parameters
        ----------
        b : bool
        """
        self.patch.set_visible(b)
        self.stale = True


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
                ax=ax, fontsize=15, yerr='errors', color='#FBE122', ecolor = '#FBE122')
    #set title and label 
    plt.title('Coefficients of Features in With 95% Confidence Interval', fontsize=20)
    ax.set_ylabel('Coefficients', fontsize=15)
    ax.set_xlabel(' ')
    #coefficients 
    ax.scatter(x= np.arange(coef_df.shape[0]),
              marker='+', s=50, 
              y=coef_df['coef'], color='#FBE122')
    plt.legend(fontsize= 15,frameon=True, fancybox=True, facecolor='black')
    set_frameon(ax, False)
    set_frameon(fig, False)
    return plt.show()


def make_ols(df, x_columns,target='price'):
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
    fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='r', color='y', alpha=.65, fit=True, markerfacecolor="#FBE122")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    fig = set_frameon(fig, False)
    #return model for later use 
    return model


def get_percentile(data_n_col):
    """Print out all of your percentiles for a given column in a dataframe
       Example: data['price'] """
    for i in range(1,100):
        q = i / 100
        print('{} percentile: {}'.format(q, data_n_col.quantile(q=q)))


def quadregplot(model, column):
    """Pass in model and column to stats model to create 4 regression plots in Seattle Storm colors"""
    fig = plt.figure(figsize=(15,8))
    fig = sm.graphics.plot_regress_exog(model, column, fig=fig)
    ax1, ax2, ax3, ax4 = fig.get_axes()
    ax1.properties()['children'][0].set_color('#A2AAAD')
    ax1.properties()['children'][1].set_color('#FBE122')
    ax1.properties()['children'][2].set_color('#2C5234')
    ax2.properties()['children'][0].set_color('#2C5234')
    ax2.properties()['children'][1].set_color('#FBE122')
    ax3.properties()['children'][1].set_color('#FBE122')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    green_patch = mpatches.Patch(color='#FBE122', label='Price')
    yellow_patch = mpatches.Patch(color='#2C5234', label='Fitted')
    ax1.legend(handles=[green_patch, yellow_patch])


def reformat_large_ticker_values(ticker_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    Code Sourced from "https://dfrieds.com/data-visualizations/how-format-large-tick-values.html"
    """
    if ticker_val >= 1000000000:
        val = round(ticker_val/1000000000, 1)
        new_ticker_format = '{:}B'.format(val)
    elif ticker_val >= 1000000:
        val = round(ticker_val/1000000, 1)
        new_ticker_format = '{:}M'.format(val)
    elif ticker_val >= 1000:
        val = round(ticker_val/1000, 1)
        new_ticker_format = '{:}K'.format(val)
    elif ticker_val < 1000:
        new_ticker_format = round(ticker_val, 1)
    else:
        new_ticker_format = ticker_val
    # make new_tick_format into a string value
    new_ticker_format = str(new_ticker_format)
    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_ticker_format.find(".")
    if index_of_decimal get_ipython().getoutput("= -1:")
        value_after_decimal = new_ticker_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_ticker_format = new_ticker_format[0:index_of_decimal] + new_ticker_format[index_of_decimal+2:]
    return new_ticker_format


#wrote up our data types to save on computer space and stop some of them from being inccorectly read as objs
kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 
             'floors': float, 'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 
             'yr_built': int, 'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float}


# Data provided from Flatiron 
kc_data = pd.read_csv(r'~\Documents\Flatiron\pro2\data\kc_house_data.csv', parse_dates = ['date'], dtype=kc_dtypes)
# Data gathered from ArcGis
schools = pd.read_csv(r'~\Documents\Flatiron\pro2\data\Schools.csv')
foods = pd.read_csv(r'~\Documents\Flatiron\pro2\foods.csv')


# Removing latitudes and longitudes that couldn't be found from adresses in ArcGis data 
foods = foods.loc[foods['lat'] get_ipython().getoutput("= '[0.0]'].copy()")
foods = foods.loc[foods['long'] get_ipython().getoutput("= '[0.0]'].copy()")
# Convert latitude and longitude into floats so we can use them to calculate distance 
foods['lat'] = foods['lat'].astype(dtype=float)
foods['long'] = foods['long'].astype(dtype=float)


# Seperate grocery stores from restaurants
rest = foods.loc[foods['SEAT_CAP'] get_ipython().getoutput("= 'Grocery']")
groc = foods.loc[foods['SEAT_CAP'] == 'Grocery']


# we can't append these directly into a DataFrame very easily, so we're going to start an empty dictionary 
# that's going to store all of our values and be converted into a dataframe later on 
kc_dict = {}


i = 0
# here we're going to iterate over the latitude and longitude of each house, and calculate the 
# distance between our housing and our other dataframes 
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


# let's turn our dictionary into a dataframe we can work with 
kc = pd.DataFrame.from_dict(kc_dict, orient='index')
# it will be the same length as our dataframe and in the same order, so we're going to merge it on index
kc_data = kc_data.merge(kc, left_index=True, right_index=True)


# though nice in dictionary form, we shouldnt have spaces in our names for our dataframes so we'll
# be renaming them here, but keeping the dictionary in case anyone needs to reference what each name means 
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


# looking at a histogram of value counts for al of our data can give us a sense of how it's 
#distributed and what columns we might have issues with
hist = kc_data[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 
                'lat', 'long', 'sqft_living15', 'sqft_lot15', 'mi_2_scl', 'scls_in_mi', 'mi_2_rest',
                'rest_in_mi', 'mi_2_groc', 'groc_in_mi']]

hist.hist(figsize=(15,15), color='#2C5234')
plt.tight_layout()


# a scatter matrix will compare all of our columns against eachother, it's very large and takes a while 
# with so much data, but can be really informative
scatter = kc_data[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 
                'lat', 'long', 'sqft_living15', 'sqft_lot15', 'mi_2_scl', 'scls_in_mi', 'mi_2_rest',
                'rest_in_mi', 'mi_2_groc', 'groc_in_mi']]
fig = pd.plotting.scatter_matrix(scatter,figsize=(20,20));


#a heatmap will help visualize the multicolinearity in our data and help us to see if anything stands out l
fig, ax = plt.subplots(figsize=(25,20))
corr = kc_data.corr().abs().round(3)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr, annot=True, mask=mask, cmap='YlGn', ax=ax)
plt.setp(ax.get_xticklabels(), 
         rotation=45, 
         ha="right",
         rotation_mode="anchor")
ax.set_title('Correlations')
set_frameon(ax, True)
fig.tight_layout()


kc_data['sqft_basement'] = kc_data['sqft_basement'].map(lambda x :  1 if x == 0 else x )


#getting rid of multicolinearity in sqftage 
kc_data['sqft_total'] = kc_data['sqft_living']*kc_data['sqft_lot']
kc_data['sqft_neighb'] = kc_data['sqft_living15']*kc_data['sqft_lot15']
kc_data['sqft_habitable'] = kc_data['sqft_above']*kc_data['sqft_basement']


#print columns we will be using going forward 
kc_data.columns


#make a copy of the dataframe holding only columns we'll be including
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


#seperating our data into different income brackets, as a 100,000 house is unlikely to be helpful
#in predicting the price of a 1,000,000 house 
hightier = kc_data[kc_data.price >800000]
midtier = kc_data[(kc_data.price > 300001) & (kc_data.price<=800000) ]
lowtier = kc_data[kc_data.price <=300000]


#as we go through we will notice that some features apply to different income brackets, 
#so seperating them out helps us choose the features that best apply to each of them 

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


#putting all of our price brackets together to go into our model 
price_tiers = [('high', hightier, highincome),
               ('mid', midtier, mediumincome),
               ('low', lowtier, lowincome)]


#using the model function we defined earlier to model and plot our qq plots for each income bracket 
for name, tier, income in price_tiers:
    print(name.upper())
    make_ols(tier, income)


# we're going to normalize price a bit by filtering out all of our homes more than or less than 2 
# standard deviations from our mean housing price 
for col in ['price']:
    col_zscore = str(col + '_zscore')
    kc_data[col_zscore] = (kc_data[col] - kc_data[col].mean())/kc_data[col].std()
    kc_data = kc_data.loc[kc_data[col_zscore] < 2]
    kc_data = kc_data.loc[kc_data[col_zscore] > (-2)]
    #we're also going to drop our z scored price, after exploring with it for a while 
    #we found the regular price much more helpful
    kc_data = kc_data.drop(col_zscore, axis = 1)


# taking a look at our prices to make sure they're normal enough for us to use 
with plt.style.context('dark_background'):
    plt.figure(figsize=(15,4))
    plt.plot(kc_data['price'].value_counts().sort_index(), color='#FBE122')
# looks good!


# checking out our percentiles to see if there's anything weird we can see 
get_percentile(kc_data['price'])


#taking a quick peak at our minumumns, maximums & other data provided by a .describe() we can easily see some outliers
kc_data.describe()


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


# after quite a bit of modeling, these came down to our best price ranges per income bracket 
hightier = kc_data[(kc_data.price >= 640000) & (kc_data.price <= 900000)]
uppermidtier = kc_data[(kc_data.price >= 480000) & (kc_data.price <= 640000) ]
midtier = kc_data[(kc_data.price >= 348000) & (kc_data.price <= 480000) ]
lowtier = kc_data[(kc_data.price >= 210000) & (kc_data.price <= 348000) ]


# these are the features for each income bracket that have significant p-values & low correlation scores
# that help us produce the best fit model 
highincome = ['bathrooms', 'floors', 'sqft_neighb', 
              'sqft_habitable', 'thru2020',
              'zip006t011', 'zip107t115',
              'zip116t122', 'zip177t199', 
              'mi_2_scl', 'scls_in_mi', 'mi_2_rest',
              'mi_2_groc', 'groc_in_mi']

uppermedincome = ['bathrooms',  'lat', 'sqft_habitable',   
                  'C', 'Bmin', 'B', 
                  'zip014t024', 'zip027t031', 'zip032t039', 
                  'zip070t077', 'zip125t144', 'zip146t168', 
                  'thru2000', 'thru2020', 'thru60', 'thru80']

mediumincome = ['bathrooms',  'lat', 'long', 
                'sqft_habitable', 'view2',   
                'Cpl', 'Bmin', 'B', 'Bpl',   
                'zip006t011', 'zip014t024', 'zip032t039', 
                'zip055t065', 'zip070t077', 'zip092t106', 
                'zip177t199', 'rest_in_mi', 'groc_in_mi',
                'thru2000', 'thru2020', 'thru60', 'thru80']

lowincome = ['bathrooms', 'waterfront', 'lat', 'long',
             'sqft_total', 'sqft_habitable', 
             'view1', 'view2', 'view3', 
             'C', 'Cpl', 'Bmin', 'B',
             'zip040t053', 'zip055t065', 'zip092t106', 
             'zip107t115', 'zip146t168', 
             'groc_in_mi']


# since we added another price bracket we need to redefine our price tiers 
price_tiers = [('high', hightier, highincome),
               ('upmid', uppermidtier, uppermedincome),
               ('mid', midtier, mediumincome),
               ('low', lowtier, lowincome)]


# getting our final model, printing them in income order followed by their qq-plots 
for name, tier, income in price_tiers:
    print(name.upper())
    make_ols(tier, income)


#first step is to seperate out the data we're going to use for this model
high_data = hightier[['price', 'bathrooms', 'floors', 'sqft_neighb', 
                      'sqft_habitable', 'thru2020',
                      'zip006t011', 'zip107t115',
                      'zip116t122', 'zip177t199', 
                      'mi_2_scl', 'scls_in_mi', 'mi_2_rest',
                      'mi_2_groc', 'groc_in_mi']].copy()
# splitting it into 25/75 training/testing data to make sure our model is consistent 
training_data, testing_data = train_test_split(high_data, test_size=0.25, random_state=44)


#split columns
target = 'price'
predictive_cols = training_data.drop('price', 1).columns


#assign model a name so we can call on it to plot later on 
high_model = make_ols(hightier, predictive_cols)


#print and take a look at our coefficients 
high_model.params.sort_values()


# assign your predictions 
y_pred_train = high_model.predict(training_data[predictive_cols])
y_pred_test = high_model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)


#calculating MSE and converting it to $ 
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)
print('Training Error: $', sqrt(train_mse), '\nTesting Error:', sqrt(test_mse))


#plotting our coefficients 
plotcoef(high_model)


sns.lmplot(data=high_data, x='floors',y='price', palette=['#fbe122'], markers='+')
plt.xlabel('Floors',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("Floors/Lofts & Price for Upper")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
plt.xlim(.75, 3.5)
plt.ylim(620000, 920000)
plt.show()


sns.lmplot(data=high_data, x='bathrooms',y='price',palette=['#FBE122', '#C9BF26', '#939A2B', '#62772F', '#44644C', '#657C6E', '#84938E', '#A2AAAD'])
plt.xlabel('Bathrooms',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("Bathrooms & Price for Upper")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
plt.xlim(0.75, 5.5)
plt.show()


sns.lmplot(data=high_data, x='floors',y='price',palette=[])
plt.xlabel('Floors',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("Floors/Lofts & Price for Upper")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
plt.show()


quadregplot(high_model, 'thru2020')


#first step is to seperate out the data we're going to use for this model
upper_med_data = uppermidtier[['bathrooms',  'lat', 'sqft_habitable',   
                               'C', 'Bmin', 'B', 'price',
                               'zip014t024', 'zip027t031', 'zip032t039', 
                               'zip070t077', 'zip125t144', 'zip146t168', 
                               'thru2000', 'thru2020', 'thru60', 'thru80']].copy()
# splitting it into 25/75 training/testing data to make sure our model is consistent 
training_data, testing_data = train_test_split(upper_med_data,test_size=0.30, random_state=55)



#split columns
target = 'price'
predictive_cols = training_data.drop('price', 1).columns


#assign model a name so we can call on it to plot later on 
uppmid_model = make_ols(training_data, predictive_cols)


#print and take a look at our coefficients 
uppmid_model.params.sort_values()


# assign your predictions 
y_pred_train = uppmid_model.predict(training_data[predictive_cols])
y_pred_test = uppmid_model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)


#calculating MSE and converting it to $ 
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)
print('Training Error: $', sqrt(train_mse), '\nTesting Error:', sqrt(test_mse))


#plotting our coefficients 
plotcoef(uppmid_model)


sns.violinplot(data=upper_med_data, x='B',y='price',palette=['#2C5234', '#FBE122'])
plt.xlabel('B Rating',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("B Rating Price Value Upper Middle")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
plt.xticks(ticks=[0,1], labels=['Not B', 'B'])
plt.show()


sns.violinplot(data=upper_med_data, x='Bmin',y='price',palette=['#2C5234', '#FBE122'])
plt.xlabel('Bmin Rating',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("Bmin Rating Price Value Upper Middle")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
plt.xticks(ticks=[0,1], labels=['Not B', 'B'])
plt.show()


sns.violinplot(data=upper_med_data, x='C',y='price',palette=['#2C5234', '#FBE122'])
plt.xlabel('C Rating',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("C Rating Price Value Upper Midde")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
plt.xticks(ticks=[0,1], labels=['Not C', 'C'])
plt.show()


sns.lmplot(data=upper_med_data, x='bathrooms',y='price',palette=[])
plt.xlabel('Bathrooms',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("Bathrooms & Price for Upper Middle")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
plt.xlim(.85, 5.25)
plt.show()


#first step is to seperate out the data we're going to use for this model
mid_data = midtier[['bathrooms',  'lat', 'long', 
                    'sqft_habitable', 'view2', 'price', 
                    'Cpl', 'Bmin', 'B', 'Bpl',   
                    'zip006t011', 'zip014t024', 'zip032t039', 
                    'zip055t065', 'zip070t077', 'zip092t106', 
                    'zip177t199', 'rest_in_mi', 'groc_in_mi',
                    'thru2000', 'thru2020', 'thru60', 'thru80']].copy()
# splitting it into 25/75 training/testing data to make sure our model is consistent 
training_data, testing_data = train_test_split(mid_data, test_size=0.30, random_state=70)


#split columns
target = 'price'
predictive_cols = training_data.drop('price', 1).columns


#assign model a name so we can call on it to plot later on 
mid_model = make_ols(mid_data, predictive_cols)


#print and take a look at our coefficients 
mid_model.params.sort_values()


# assign your preditions 
y_pred_train = mid_model.predict(training_data[predictive_cols])
y_pred_test = mid_model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)


#calculating MSE and converting it to $ 
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)
print('Training Error: $', sqrt(train_mse), '\nTesting Error:', sqrt(test_mse))


#plotting our coefficients 
plotcoef(mid_model)


plt.figure(figsize=(5,8))
sns.lineplot(data=mid_data, x='Bpl',y='price',color='#FBE122', alpha=.50)
plt.xlabel('Grade',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("Grade Vs. Home Value: Middle Income")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
sns.lineplot(data=mid_data, x='B',y='price',color='#A2AAAD', ax=ax, alpha=0.5)
sns.lineplot(data=mid_data, x='Bmin',y='price',color='#151f17', ax=ax)
plt.ylim(410000, 470000)
green_patch = mpatches.Patch(color='#baa823', label='B Plus')
yellow_patch = mpatches.Patch(color='#A2AAAD', label='B')
grey_patch = mpatches.Patch(color='#151f17', label='B Minus')
ax.legend(handles=[green_patch, yellow_patch, grey_patch], labels=['B Plus', 'B', 'B Minus'], loc='upper left')
plt.xticks(ticks=[0.15,.95], labels=['Not B', 'B'])
plt.show()


plt.figure(figsize=(5,8))
sns.lineplot(data=mid_data, x='zip014t024',y='price',color='#FBE122', alpha=.50)
plt.xlabel('Zipcodes',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("Suburban Zipcodes Vs. Home Value: Middle Income")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
sns.lineplot(data=mid_data, x='zip092t106',y='price',color='#151f17', ax=ax, alpha=1)
sns.lineplot(data=mid_data, x='zip055t065',y='price',color='#A2AAAD', ax=ax)
sns.lineplot(data=mid_data, x='zip177t199',y='price',color='#ffffff', ax=ax, alpha=1)
plt.ylim(385000, 425000)
plt.xlim(-.15, 1.15)
plt.xticks(ticks=[0.15,.95], labels=['Not Suburban', 'Suburbs'])
plt.show()


sns.violinplot(data=mid_data, x='Bmin',y='price',palette=['#2C5234', '#FBE122'])
plt.xlabel('B Minus Rating',labelpad=16)
plt.ylabel('Price',labelpad=16)
plt.title("B Minus Rating Price Value Middle")
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(reformat_large_ticker_values));
plt.xticks(ticks=[0,1], labels=['Not B', 'B'])
plt.show()


#first step is to seperate out the data we're going to use for this model
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


#assign model a name so we can call on it to plot later on 
low_model = make_ols(low_data, predictive_cols)


#print and take a look at our coefficients 
low_model.params.sort_values()


# assign your predictions 
y_pred_train = low_model.predict(training_data[predictive_cols])
y_pred_test = low_model.predict(testing_data[predictive_cols])
# then get the scores:
train_mse = mean_squared_error(training_data[target], y_pred_train)
test_mse = mean_squared_error(testing_data[target], y_pred_test)


#calculating MSE and converting it to $ 
print('Training MSE:', train_mse, '\nTesting MSE:', test_mse)
print('Training Error: $', round(sqrt(train_mse), 2), '\nTesting Error: $', round(sqrt(test_mse), 2))


#plotting our coefficients 
plotcoef(low_model)


quadregplot(low_model, 'lat')
#as you can see here, the farther west, towards the cities, you go, the more expensive homes become. 
#please note we are not missing data in those gaps, those represent bodies of water where few people live on small islands or in house boats 


quadregplot(low_model,'bathrooms')
# 6,000$ doesn't look like much compared to prices in the 450,000, but, an increase from 1 to 4 could add over 20k and possibly bump you up to a higher 
# grade, making your home worth even more. 


#Bathrooms High
fig = plt.figure(figsize=(10,10))
sns.boxplot(x='bathrooms',y='price',data=low_data,color='#2C5234', palette=['#fbe122', '#c9bf26', '#939a2b', '#62772f', '#44644c', '#657c6e', '#84938e', '#a2aaad'])
plt.title('Number of Bathrooms Vs. Price in Lower Income Homes')
plt.xlabel('Number of Bathrooms in Home')
plt.xlim(-.5, 9.5)
plt.tight_layout()
