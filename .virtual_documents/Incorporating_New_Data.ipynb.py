import pandas as pd 
import numpy as np


kc_dtypes = {'id': str, 'date' : str,  'price': float, 'bedrooms' : int, 'bathrooms' : float, 'sqft_living': int, 'sqft_lot': int, 'floors': float, 
             'waterfront': float, 'view' : float, 'condition': float, 'grade': int, 'sqft_above': int, 'yr_built': int,
             'yr_renovated': float, 'zipcode': float, 'lat': float, 'long': float, 'sqft_living15': int, 'sqft_lot15': int}

kc_data = pd.read_csv('dsc-phase-2-project/data/kc_house_data.csv', parse_dates = ['date'])


kc_parcels = pd.read_csv('dsc-phase-2-project/data/EXTR_Parcel.csv', encoding='ISO-8859-1')

i = 0
j = 0

kc_parcels['Major'] = kc_parcels['Major'].astype(str)
kc_parcels['Major'] = kc_parcels['Major'].str.strip()    
kc_parcels['Minor'] = kc_parcels['Minor'].astype(str)
kc_parcels['Minor'] = kc_parcels['Minor'].str.strip()

for row in kc_parcels:
    while len(str(kc_parcels['Minor'][i])) + j < 4:
             kc_parcels['Major'][i]*10
             j += 1
             
    i += 1
    
kc_parcels['id'] = kc_parcels['Major'] + kc_parcels['Minor']


kc_data.merge(kc_parcels, how='inner', left_on='id')


kc_data.loc[kc_data['id'] ==  162405162405]


kc_parcels['id'] = kc_parcels[i]



