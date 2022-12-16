# Colorado Landsliding

This repository contains Python code used for EDA and nonspatial classification analysis of observed landslide
points with susceptibility factors joined. The spatial classification and prediction of landslide points
and classification and prediction of landslide polygons were done in ArcGIS Pro using forest-based classification
and regression as detailed in the paper.

The shapefile that is being read into this analysis originates from the USGS Landslide Inventory, where observed
landslides based on previous publications in Colorado were compiled by the Colorado Geological Survey and requested
and downloaded for use in this project. The twelve landslide susceptibility controlling factors as detailed in the 
Data section were requested and downloaded, preprocessed to normalize units, ensure a constant spatial reference
system and raster resolution, mosaicked and combined data in separate regions into layers covering the entire
study area, and converted to the same file type to ensure uniformity in analysis. The resulting preprocessed
layers were spatially joined to the observed landslide points shapefile. Further details are included in the paper.

The purpose of this nonspatial classification analysis of landslide points is to determine whether it is feasible
to construct a reasonably accurate classification model to predict the severity of observed landslides. If several
models are feasible, this code assists in choosing which model has the best performance. Because spatial classification
requires intensive computational resources and time, only one model can be run in spatial analysis. This nonspatial
analysis serves as a precursor to the spatial analysis and prediction of landslide susceptibility that forms the core of
the project and deliverables.

## pts-eda.py

This code reads the shapefile of observed point landslides with feature data joined to each point.
The code then provides some exploratory data analysis by displaying the frequency of each controlling factor and
their relationship with the target variable Confidence. 

## model-app.py

This code reads the shapefile of observed point landslides with feature data joined to each point.
The code then conducts a train-test split and trains a classifier on the data (multiple classifiers were tested,
the given version uses random forest). The code then provides a suite a functions for evaluating classifier performance,
including regular and normalized confusion matrices, feature importances, precision-recall and receiver operating
characteristic curves, and cross-validation.

## US_Landslide_Points_Project.shp

The shapefile that is being used by the Python code, with all of the preprocessing steps and feature data joins 
conducted as discussed above. "Project" specifies that these points have been projected into the NAD 1983 (2011)
StatePlane Colorado North FIPS 0501 (Meters) spatial reference system to distinguish it from previous versions
of the shapefile in various steps of preprocessing (not included in this repository).

## SML312_Final_Paper.pdf

The final paper.
