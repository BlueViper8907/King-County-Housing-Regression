import pandas as pd
import arcgis
from arcgis import *
from arcgis.features import GeoAccessor,GeoSeriesAccessor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from arcgis.mapping import WebMap
from IPython.display import display


df = pd.read_csv('data/kc_house_data.csv')



gis = GIS("http://www.arcgis.com/", "shadowsword_0","Acidblade1")


def gis_content():
    item = gis.content.get('ff9e4774ee8641f48cccac97dca753db#data')
    flayer = item.layers[0]
    food = flayer.query(where = "ESITE = '0000000'").sdf.copy()
    item = gis.content.get('3f263039314d44cc93384fe1f4229796#data')
    flayer = item.layers[0]
    public = flayer.query(where = "ZIPCODE > 0").sdf.copy()
    
    item = gis.content.get('0dfe37d2a68545a699b999804354dacf')
    flayer = item.layers[0]
    private = flayer.query(where = "STATE = 'WA'").sdf.copy()
    
    #Editing data for later use
    public_df = public.drop(['FID','PIN','SCH_CLASS','CODE','FEATURE_ID','ESITE','MAJOR','MINOR','SHAPE'],1)
    private_df = private.drop(['FID','NCESID','ZIP4','VAL_METHOD','VAL_DATE','SHELTER_ID','COUNTRY','SOURCE','SOURCEDATE','NAICS_CODE','FT_TEACHER','START_GRAD','END_GRADE','COUNTYFIPS','SHAPE','TYPE','STATUS','VAL_METHOD','VAL_DATE'],1)
    food_df = food.drop(['RECORD_ID','FACILITY_NAME','CHAIN_NAME','CHAIN_ESTABLISHMENT','SITE_ADDRESS','ABB_NAME','ESITE','FEATURE_ID'],1)
#     private_df = private_df.rename(columns ={'ZIP':'zipcode','LONGITUDE':'long','LATITUDE':'lat'})
#     public_df = public_df.rename(columns = {'LONG_CEN': 'long', 'LAT_CEN': 'lat','ZIPCODE':'zipcode'})
    private_df['Private'] = 'yes'
    public_df['Private'] = 'No'
#     public_df['zipcode'] = public_df['zipcode'].astype(str).astype(int)
#     private_df['zipcode'] = private_df['zipcode'].astype(str).astype(int)
    public_df['Private'].astype(str)
    
    return (food_df,public_df,private_df) 



food,public,private = gis_content()



def mapping(df):
    """Taking in a dataframe to assign the budget through all fields. Takes schools into account as well. If not looking for schools
        just enter through"""
#     mapp = gis.content.get('579a5ea9a24d4b85b237ef8e9cb578b4')
    budget = int(input('What is your budget: '))
    budget_df = df.loc[df.price<=budget].copy()
    budget_df.is_copy = None
    pdx_map = gis.map('Seattle WA')
    pdx_map.basemap = 'streets-night-vector'
    schools = input('Are you interested in schools? ')
    if (schools == 'Yes') or (schools == 'yes') and (budget is not None):
        school_sdf = pd.DataFrame.spatial.from_xy(private,'LONGITUDE','LATITUDE')
        school_sdf.spatial.plot(map_widget = pdx_map,renderer_type = "c",symbol_type = 'simple',symbol_style = 'x',title = 'Schools', 
                    col='name',
                    cmap='winter',  # matplotlib color map
                    alpha=0.7,
                    size = .5, 
                    outline_color=[0,0,0,0])
#         interest = input('Private school,Public school or both? ')
#         if (interest =='Private') or (interest =='private'):
            
            
#             school_sdf = pd.DataFrame.spatial.from_xy(private,'long','lat')
#             school_sdf.spatial.plot(map_widget = pdx_map,renderer_type = "c",symbol_type = 'simple',symbol_style = 'o',title = 'Schools', 
#                     col='zipcode',
#                     cmap='winter',  # matplotlib color map
#                     alpha=0.7,
#                     size = .5, 
#                     outline_color=[0,0,0,0])
#         elif(interest =='Public') or (interest == 'public'):
# #           
#             school_sdf = pd.DataFrame.spatial.from_xy(public,'long','lat')
#             school_sdf.spatial.plot(map_widget = pdx_map, renderer_type = "c",symbol_type = 'simple',symbol_style = 'o',title = 'Public', 
#                     col='zipcode',
#                     cmap='winter',  # matplotlib color map
#                     alpha=0.7,
#                     size = .5, 
#                     outline_color=[0,0,0,0])
#         else:
#             school_sdf = pd.DataFrame.spatial.from_xy(all_schools,'LONGITUDE','LATITUDE')
#             school_sdf.spatial.plot(map_widget = pdx_map,renderer_type = "c",symbol_type = 'simple',symbol_style = 'o',title = 'Schools', 
#                       col='zipcode',
#                       cmap='winter',  # matplotlib color map
#                     alpha=0.7,
#                     size = .5, 
#                     outline_color=[0,0,0,0])
        
       
    data_sdf = pd.DataFrame.spatial.from_xy(budget_df, 'long','lat')
    
       
    data_sdf.spatial.plot(map_widget=pdx_map, renderer_type = "c",marker_size = 5
                    ,symbol_type = 'simple',symbol_style='d',
                    title='Pricing of houses',
                    col='price',
                    cmap='spring',  # matplotlib color map
                    alpha=0.7,
                    size = .5,
                    outline_color=[0,0,0,0]
                     )
#     food_sdf = pd.DataFrame.spatial(food)
#     food_sdf.plot(map_widget = pdx_map,renderer_type = "c",symbol_type = 'simple', symbol_style = 'x',
#                   title='Pricing of houses',
#                     col='price',  
#                   cmap='spring',  # matplotlib color map
#                     alpha=0.7,
#                     size = .5,
#                     outline_color=[0,0,0,0])
    pdx_map.legend = True
    display(pdx_map)
#     display(mapp)
# took legend out since the unique renderer_type makes it hard to assign general information through it. 
# Will work on getting that working so the distance from houses shows schools locally by zipcode.
# sns.barplot(x = 'price',y = 'sqft_living',data = budget_df)


mapping(df)


private.head()


public.head()





# pull data from https://blue.kingcounty.com/Assessor/eRealProperty/Detail.aspx?ParcelNbr=
# adding the id number from each of the dataframes into the last bit of the code adding that to a new dataframe to work with the code


test_map = gis.content.get('579a5ea9a24d4b85b237ef8e9cb578b4')

item = gis.content.search('kc_house_data')

# sdf = pd.DataFrame.spatial.from_layer(item[0])


online_map = WebMap(test_map)
housing = online_map.layers[0]
housing_df = pd.DataFrame.spatial.from_layer(housing)
# for layer in online_map.layers:
    
#     print(layer.title)


online_map


housing



