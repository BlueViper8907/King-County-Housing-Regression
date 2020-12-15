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


df = pd.read_csv('data/Cleaned_Dataset.csv')



gis = GIS("http://www.arcgis.com/", "shadowsword_0","Acidblade1")


lowtier = df[(df.price >= 210000) & (df.price <= 348000) ].copy()
midtier = df[(df.price >= 348000) & (df.price <= 480000) ].copy()
uppermidtier = df[(df.price >= 480000) & (df.price <= 640000) ].copy()
hightier = df[(df.price >= 640000) & (df.price <= 900000)].copy()


def gis_content():
    item = gis.content.get('ff9e4774ee8641f48cccac97dca753db#data')
    food = item.layers[0]
    item = gis.content.get('3f263039314d44cc93384fe1f4229796#data')
    flayer = item.layers[0]
    public = flayer.query(where = "ZIPCODE > 0")
    item = gis.content.get('d4a439bcf5d54e5f80cde3285d0cf3cd')
    dataset = item.layers[0]
    item = gis.content.get('175728366bb24060904323678963c60e')
    flayer = item.layers[0]
    private = flayer.query()
    return (private,public,food) 


def Cleaning(dataframe):
    data = dataframe.drop(['id','date', 'waterfront', 'view','condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15',
              'sqft_lot15'],1)
    return data


def mapping(budget_df):
    """Taking in a dataframe to assign the budget through all fields. Takes schools into account as well. If not looking for schools
        just enter through"""
    private,public,food = gis_content()

    pdx_map = gis.map('Seattle, WA')
    pdx_map.basemap = 'streets-night-vector'
    schools = input('Are you interested in schools? ')
    if (schools == 'Yes') or (schools == 'yes') and (budget_df is not None):
        interest = input('Private school or Public school: ')
        if (interest =='Private') or (interest =='private'):
             pdx_map.add_layer(private)
        elif(interest =='Public') or (interest == 'public'):
            pdx_map.add_layer(public)       
       
    data_sdf = pd.DataFrame.spatial.from_xy(budget_df, 'long','lat')
    data = Cleaning(data_sdf)
       
    data.spatial.plot(map_widget=pdx_map, renderer_type = "c",marker_size = 5,
                    symbol_type = 'simple',symbol_style='d',
                    title='Pricing of houses',
                    col='price',
                    cmap='winter',  # matplotlib color map
                    alpha=0.7,
                    size = .5,
                    outline_color=[0,0,0,0]
                     )
    pdx_map.add_layer(food)
   
    display(pdx_map)



mapping(df)


mapping(lowtier)


mapping(midtier)


mapping(uppermidtier)
























