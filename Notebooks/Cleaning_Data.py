
import pandas as pd
import numpy as np
import scipy
import sklearn
import datetime
from scipy import stats

def data_set():
    kc_columns = ['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
              'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
              'sqft_lot15']

    kc_dtypes = {'id': int, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int,'floors': float,
             'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 'yr_built': int,
             'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float, 'sqft_living15': int, 'sqft_lot15': int}

    kc_data = pd.read_csv('dsc-phase-2-project/data/kc_house_data.csv', dtype = kc_dtypes, parse_dates = ['date'])

    kc_data['sqft_basement'] = kc_data['sqft_basement'].replace({'?': 0})
    kc_data['sqft_basement'] = kc_data['sqft_basement'].astype(dtype=float, errors='ignore')
    return kc_data
df = data_set()
def Cleaning(kc_data):
    #look for outliers, in bedrooms, we can clearly see a single outlier, for other columns, filtering by z score will be easiest
    kc_data[kc_data['bedrooms'] == 33]
    # wouldn't be realistic for a house with 33 bedrooms to only have a sqft_living of 1620 and only 1 3/4 bathrooms so it looks like a typo
    # will adjust to 3
    kc_data[kc_data['bedrooms'] == 33] = kc_data[kc_data['bedrooms'] == 33].replace(33,3)



    #setting waterfront NaN values equal to the ratio of waterfront/non-waterfront properties, will want to try and narrow by zipcode
    #filling NaN with easily seperatable/changable values helpful
    kc_data['waterfront'] = kc_data['waterfront'].fillna(146/19221)
    kc_data['view'] = kc_data['view'].fillna((957 + (508*2) + (330*3) + (317*4))/21534)
    kc_data['yr_renovated'] = kc_data['yr_renovated'].fillna(0)



    #here are the columns i think it's reasonable to filter by z score, filtering these 3 removes about 100 rows
    find_outliers = ['sqft_basement', 'sqft_above', 'price']

    for column in find_outliers:
        z_score = stats.zscore(kc_data[[column]])
        abs_z_score = np.abs(z_score)
        filtered_entries = (abs_z_score < 3).all(axis=1)
        kc_data[str(column + '_z')] = kc_data[column][filtered_entries]
        kc_data = kc_data.dropna(axis=0)



    #Convert to integer for whole number year
    kc_data['yr_renovated'] = kc_data['yr_renovated'].astype('int')



    #check for duplicates
    duplicates = kc_data[kc_data.duplicated()]
    print(len(duplicates))
    display(duplicates.head())
    #none found! Now that we are free of outliers, duplicates, null values, etc, we can consider our data cleaned
    


    kc_data = kc_data.drop(['sqft_basement_z','sqft_above_z','price_z'], 1)



    kc_data.to_csv('data/Cleaned_Dataset.csv', index=False)
