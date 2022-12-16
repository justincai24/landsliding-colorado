import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

""" Author: Justin Cai for SML312 Research Projects in Data Science
Description: This code reads the shapefile of observed point landslides with feature data joined to each point.
The code then conducts a train-test split and trains a classifier on the data (multiple classifiers were tested,
the version below uses random forest). The code then provides a suite a functions for evaluating classifier performance,
including regular and normalized confusion matrices, feature importances, precision-recall and roc curves, and cross-validation.

The sources of landslide and feature data and the preprocessing steps necessary to keep important qualities like spatial reference
and raster resolution constant were done in ArcGIS Pro and described in the paper. The shapefile being read in this code is
the result of preprocessing of all landslide susceptibility factor data and conducting a spatial join of each susceptibility
factor to the landslide points.

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

# Used for testing binary classification of landslides, mainly for debugging purposes
#pts['target'] = [0 if x==0 else 1 for x in pts['Confidence']]

# Feature names list
feature_names = ['Geology','Soil','Aspect','NDVI','Road_Dist','Fault_Dist','Land_Cover','Water_Dist','Slope','Elevation','Plan_Cu','Profile_Cu']
n_classes = len(feature_names)

# Define target and feature classes and conduct train-test split
# The binarize line is used if we want to construct multiclass precision-recall and roc curves
X = pts[feature_names]
#y = label_binarize(pts['Confidence'], classes=[*range(n_classes)])
y = pts['Confidence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Construct classifier (the clf variable was interchanged with each classifier and parameters described in the paper)
# The OneVsRest and y_score lines are used if we want to construct multiclass precision-recall and roc curves
#clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5))
clf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#y_score = clf.predict_proba(X_test)

# Displays confusion matrix of predicted results
def disp_conf_mat(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Possible','Probable','Likely','Confident','Highly Confident'])
    disp.plot()
    plt.show()
    
# Displays normalized confusion matrix of predicted results
def norm_conf_mat(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(13,9))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':18}, cmap=plt.cm.OrRd, linewidths=0.2)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_test, y_pred))

# Displays cross-validation accuracy and standard deviation of model
def print_scor(y_pred, y_test, clf, X, y):
    print("Accuracy: " + str(accuracy_score(y_pred,y_test)))
    scores = cross_val_score(clf, X, y, cv=4)
    print("Cross-validation accuracy: " + str(scores.mean()))
    print("Cross-validation standard deviation: " + str(scores.std()))

# Displays feature importance by mean decrease in impurity (only for tree-based classifiers)
def disp_imp_mdi(clf):
    importances = clf.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()

# Displays permutation feature importance
def disp_imp_pm(clf, X_test, y_test):
    result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

# Displays multiclass precision-recall curve
def prec_rec_curve(n_classes, y_test, y_score):
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

# Displays multiclass receiver operating characteristic
def roc_curve(n_classes, y_test, y_score):
    fpr = dict()
    tpr = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.show()
    
print_scor(y_pred, y_test, clf, X, y)
disp_conf_mat(y_pred, y_test)
#prec_rec_curve(n_classes, y_test, y_score)