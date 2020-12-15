from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure, ColumnDataSource
from bokeh.tile_providers import get_provider, Vendors
from bokeh.palettes import PRGn, RdYlGn
from bokeh.transform import linear_cmap,factor_cmap
from bokeh.layouts import row, column
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter
import numpy as np
import pandas as pd


df = pd.read_csv('data/kc_house_data.csv')


# Define function to switch from lat/long to mercator coordinates
def x_coord(x, y):
    
    lat = x
    lon = y
    
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + 
        lat * (np.pi/180.0)/2.0)) * scale
    return (x, y)
# Define coord as tuple (lat,long)
df['coordinates'] = list(zip(df['lat'], df['long']))
# Obtain list of mercator coordinates
mercators = [x_coord(x, y) for x, y in df['coordinates'] ]


# Create mercator column in our df
df['mercator'] = mercators
# Split that column out into two separate columns - mercator_x and mercator_y
df[['mercator_x', 'mercator_y']] = df['mercator'].apply(pd.Series)


chosentile = get_provider(Vendors.STAMEN_TERRAIN_RETINA)


# Choose palette
palette = PRGn[11]


source = ColumnDataSource(data=df)


# Define color mapper - which column will define the colour of the data points
color_mapper = linear_cmap(field_name = 'Price', palette = palette, low = df['price'].min(), high = df['price'].max())


# Set tooltips - these appear when we hover over a data point in our map, very nifty and very useful
tooltips = [('Price','@price'),('Zipcode','@zipcode'),('Bedrooms','@bedrooms'),
           ('Bathroons','@bathrooms'),('Sqft Living','@sqft_living'),('Sqft Lot','@sqft_lot'),
            ('Floors','@floors')]


# Create figure
p = figure(title = 'King County House sales', x_axis_type="mercator", y_axis_type="mercator", x_axis_label = 'Longitude', y_axis_label = 'Latitude', tooltips = tooltips,plot_width=800, plot_height=800)


p.add_tile(chosentile)
# Add points using mercator coordinates
p.circle(x = 'mercator_x', y = 'mercator_y', color = color_mapper, source=source, size=3, fill_alpha = 0.7)
#Defines color bar
color_bar = ColorBar(color_mapper=color_mapper['transform'], 
                     formatter = NumeralTickFormatter(format='0.0[0000]'), 
                     label_standoff = 13, width=8, location=(0,0))
# Set color_bar location
p.add_layout(color_bar, 'right')



# Display in notebook
output_notebook()



show(p)



