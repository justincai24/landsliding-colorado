import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

""" Author: Justin Cai for SML312 Research Projects in Data Science
Description: This code reads the shapefile of observed point landslides with feature data joined to each point.
The code then provides some exploratory data analysis by displaying the frequency of each controlling factor and
their relationship with the target variable Confidence. 

The sources of landslide and feature data and the preprocessing steps necessary to keep important qualities like
spatial reference and raster resolution constant were done in ArcGIS Pro and described in the paper. The shapefile
being read in this code is the result of preprocessing of all landslide susceptibility factor data and conducting
a spatial join of each susceptibility factor to the landslide points.

The Python code includes only the nonspatial classification analysis of observed landslide points. The spatial
classification and prediction of landslide points and classification and prediction of landslide polygons were
done in ArcGIS Pro using forest-based classification and regression as detailed in the paper."""

# Use geopandas to read shapefile of landslide points into dataframe
pts = gpd.read_file('C:/Users/justi/Documents/ArcGIS/Projects/GoldenCO/US_Landslide_Inventory/US_Landslide_Points_Project.shp')

# Remove degenerate points
pts = pts.loc[pts["Road_Dist"]>0]
pts = pts.loc[pts["Water_Dist"]>0]
pts = pts.loc[pts["Fault_Dist"]>0]
pts = pts.loc[pts["Soil"]!=0]

# Display histogram of each feature
def disp_hist(pts):
    pts.hist(bins=10, figsize=(24,16), layout=(3,5))
    plt.show()

# Display binary relationship between each feature and target variable
def disp_features(features, target, pts):
    x_vars = features
    y_var = target
    g = sns.FacetGrid(pd.DataFrame(x_vars), col=0, col_wrap=3, sharex=False)
    for ax, x_var in zip(g.axes, x_vars):
        sns.scatterplot(data=pts, x=x_var, y=y_var, ax=ax)
    g.tight_layout()
    plt.show()

# Feature names list
feature_names = ['Geology','Soil','Aspect','NDVI','Road_Dist','Fault_Dist','Land_Cover','Water_Dist','Slope','Elevation','Plan_Cu','Profile_Cu']

disp_hist(pts)
disp_features(feature_names, "Confidence", pts)
