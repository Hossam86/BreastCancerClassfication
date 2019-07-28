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
from utli import print_dx_perc, variable_importance, variable_importance_plot, cross_val_metrics

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
breast_cancer = pd.read_csv(
    "BreastCancerClassfication/dataSet/wdbc.data", names=names)

dx = ["Benign", "Malignant"]

# print(breast_cancer.head())

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
fit_rf = RandomForestClassifier(n_estimators=50, random_state=42)

# ------------------------------------------------------------------------------
# Hyperparameter Optimization
# ------------------------------------------------------------------------------
np.random.seed(42)
start = time.time()
param_dist = {
    "max_depth": [2, 3, 4],
    "bootstrap": [True, False],
    "max_features": ["auto", "sqrt", "log2", None],
    "criterion": ["gini", "entropy"],
}

# cv_rf = GridSearchCV(fit_rf, cv=10, param_grid=param_dist, n_jobs=1)
# cv_rf.fit(training_set, class_set)

# print("Best Parameters using grid search: \n", cv_rf.best_params_)
end = time.time()
print("Time taken in grid search: {0: .2f}".format(end - start))

# Set best parameters given by grid search
fit_rf.set_params(criterion="gini", max_features="log2", max_depth=3)

print(fit_rf.get_params())

fit_rf.set_params(warm_start=True, oob_score=True)

# min_estimators = 15
# max_estimators = 1000
# error_rate = {}
# for i in range(min_estimators, max_estimators + 1):
#     fit_rf.set_params(n_estimators=i)
#     fit_rf.fit(training_set, class_set)

#     oob_error = 1 - fit_rf.oob_score_
#     error_rate[i] = oob_error

# # Convert dictionary to a pandas series for easy plotting
# oob_series = pd.Series(error_rate)
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_facecolor("#fafafa")
# oob_series.plot(kind="line", color="red")
# plt.axhline(0.055, color="#875FDB", linestyle="--")
# plt.axhline(0.05, color="#875FDB", linestyle="--")
# plt.xlabel("n_estimators")
# plt.ylabel("OOB Error Rate")
# plt.title("OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)")
# plt.show()

# print("OOB Error rate for 400 trees is: {0:.5f}".format(oob_series[400]))

fit_rf.set_params(n_estimators=400, bootstrap=True,
                  warm_start=False, oob_score=False)

print(fit_rf.get_params())
fit_rf.fit(training_set, class_set)
importances_rf = fit_rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]
print(importances_rf)

variable_importance(importances_rf, indices_rf, names[2:])
variable_importance_plot(importances_rf, indices_rf, names[2:])
# ------------------------------------------------------------------------------
# K-Fold Cross Validation
# ------------------------------------------------------------------------------
cross_val_metrics(fit_rf, training_set, class_set, print_results=True)
