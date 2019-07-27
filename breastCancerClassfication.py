# Ref https://www.datascience.com/resources/notebooks/random-forest-intro
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from utli import print_dx_perc

plt.style.use("ggplot")
pd.set_option("display.max_columns", 500)

# loading the data

# loading the data
names = [
    "id_number",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]
breast_cancer = pd.read_csv("DataSets/breast_cancer/wdbc.data", names=names)

dx = ["Benign", "Malignant"]

print(breast_cancer.head())

print("Here 's The dimension of our data frame,:\n", breast_cancer.shape)
print("Here 's The dimension of our columns:\n", breast_cancer.dtypes)
# ------------------------------------------------------------------------------
"""Cleaning"""
# ------------------------------------------------------------------------------
# Setting 'id_number' as our index
breast_cancer.set_index(["id_number"], inplace=True)
print(breast_cancer["diagnosis"].iloc[10])
# Converted to binary to help later on with models and plots
breast_cancer["diagnosis"] = breast_cancer["diagnosis"].map({"M": 1, "B": 0})

# ==============================================================================
# Missing Values
# ===========================================
for col in breast_cancer:
    if (breast_cancer[col].isnull().values.ravel().sum()) == 0:
        pass
    else:
        print(col)
        print((breast_cancer[col].isnull().values.ravel().sum()))

print("Sanity Check! No missing Values found!")

# ------------------------------------------------------------------------------
# Class Imbalance
# ------------------------------------------------------------------------------
print_dx_perc(breast_cancer, "diagnosis", dx)
print(breast_cancer.describe())

# ------------------------------------------------------------------------------
# Creating Training and Test Sets
# ------------------------------------------------------------------------------
feature_space = breast_cancer.iloc[:, breast_cancer.columns != "diagnosis"]
feature_class = breast_cancer.iloc[:, breast_cancer.columns == "diagnosis"]

training_set, test_set, class_set, test_class_set = train_test_split(
    feature_space, feature_class, test_size=0.20, random_state=42
)
# Cleaning test sets to avoid future warning messages
class_set = class_set.values.ravel()
test_class_set = test_class_set.values.ravel()

# ------------------------------------------------------------------------------
# Fitting Random Forest
# ------------------------------------------------------------------------------
# max_depth: The maximum splits for all trees in the forest.
# bootstrap: An indicator of whether or not we want to use bootstrap samples when building trees.
# max_features: The maximum number of features that will be used in node splitting — 
    # the main difference I previously mentioned between bagging trees and random forest. 
    # Typically, you want a value that is less than p, where p is all features in your data set.
# criterion: This is the metric used to asses the stopping criteria for the decision trees.
# Set the random state for reproductibility
fit_rf = RandomForestClassifier(random_state=42)
