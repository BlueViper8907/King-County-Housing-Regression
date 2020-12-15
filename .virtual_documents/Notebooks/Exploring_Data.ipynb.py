
    # get_ipython().getoutput("pip install arcgis")
    # get_ipython().getoutput("pip install import_ipynb")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('data/kc_house_data.csv')
private = pd.read_csv('data/PrivateSchools.csv')
public = pd.read_csv('data/PublicSchools.csv')


display(private.head())
public.head()
public = public.rename({'LONG_CEN':'LONGITUDE','LAT_CEN':'LATITUDE'})
public.head()


schools = pd.merge(public,private,)


df


display(df.isna().sum())
display(df.max())
df.min()
df.price.astype(int)



df[df['bedrooms'] == 33]
# wouldn't be realistic for a house with 33 bedrooms to only have a sqft_living of 1620 and only 1 3/4 bathrooms so it looks like a typo
# will adjust to 3 
df[df['bedrooms'] == 33] = df[df['bedrooms'] == 33].replace(33,3)
df['sqft_basement'] = df['sqft_basement'].replace('?','0').astype(float)
df['waterfront'] = df['waterfront'].fillna(0)
df['view'] = df['view'].fillna(0)
# should remove id, lat,long from dataframe to work with median 
df1 = df.drop('id',1).drop('lat',1).drop('long',1)



df.info()



df.head()





df1.median()




def gis_content():
    gis = GIS("http://www.arcgis.com/", "shadowsword_0","Acidblade1")
#     item = gis.content.get('6f6dfde35681494f92da924faf7ee47c')
#     flayer = item.layers[0]
#     common_interest = flayer.query(where = "ESITE = '0000000'").sdf.copy()
    item = gis.content.get('3f263039314d44cc93384fe1f4229796#data')
    flayer = item.layers[0]
    public = flayer.query(where = "ZIPCODE > 0").sdf.copy()
    item = gis.content.get('0dfe37d2a68545a699b999804354dacf')
    flayer = item.layers[0]
    private = flayer.query(where = "STATE = 'WA'").sdf.copy()
    
    #Editing data for later use
    public_df = public.drop(['FID','PIN','SCH_CLASS','CODE','FEATURE_ID','ESITE','MAJOR','MINOR','SHAPE'],1)
    private_df = private.drop(['FID','NCESID','ZIP4','VAL_METHOD','VAL_DATE','SHELTER_ID','COUNTRY','SOURCE','SOURCEDATE','NAICS_CODE','FT_TEACHER','START_GRAD','END_GRADE','COUNTYFIPS','SHAPE','TYPE','STATUS','VAL_METHOD','VAL_DATE'],1)
#     food_df = food.drop(['RECORD_ID','FACILITY_NAME','CHAIN_NAME','CHAIN_ESTABLISHMENT','SITE_ADDRESS','ABB_NAME','ESITE','FEATURE_ID'],1)
    private_df = private_df.rename(columns ={'ZIP':'zipcode','LATITUDE':'lat','LONGITUDE':'long'})
    public_df = public_df.rename(columns = {'LONG_CEN': 'long', 'LAT_CEN': 'lat','ZIPCODE':'zipcode'})
    private_df['Private'] = 'yes'
    public_df['Private'] = 'No'
    public_df['zipcode'] = public_df['zipcode'].astype(str).astype(int)
    private_df['zipcode'] = private_df['zipcode'].astype(str).astype(int)
    public_df['Private'].astype(str)
    return (gis,public_df,private_df) 



df.info()


gis,public,private = gis_content()






all_schools = pd.merge(public,private, how= 'outer')
display(public.head())
# food.head()
# private.head()
# all_schools.head(100)




food.head()








mapping(df)



public.head()

# zipcode = public.iloc[(public['zipcode'] == df[df['zipcode'],1],1)]
# zipcode


















