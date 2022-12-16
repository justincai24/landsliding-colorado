# Colorado Landsliding

This repository contains Python code used for EDA and nonspatial classification analysis of observed landslide
points with susceptibility factors joined. The spatial classification and prediction of landslide points
and classification and prediction of landslide polygons were done in ArcGIS Pro using forest-based classification
and regression as detailed in the paper. The data sources, downloading, and full preprocessing steps as detailed
in the model workflow in the paper were also done in ArcGIS Pro. 

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
