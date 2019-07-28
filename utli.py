import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score


def print_dx_perc(dataframe, col, dx):
    """Function used to print class distribution for our data set"""
    dx_vals = dataframe[col].value_counts()
    dx_vals = dx_vals.reset_index()

    def f(x, y): return 100 * (x / sum(y))
    for i in range(0, len(dx)):
        print(
            "{0} accounts for {1:.2f}% of the diagnosis class".format(
                dx[i], f(dx_vals[col].iloc[i], dx_vals[col])
            )
        )


def variable_importance(importance, indices, names):
    """
    Purpose:
    ----------
    Prints dependent variable names ordered from largest to smallest
    based on gini or information gain for CART model.

    Parameters:
    ----------
    names:      Name of columns included in model
    importance: Array returned from feature_importances_ for CART
                   models organized by dataframe index
    indices:    Organized index of dataframe from largest to smallest
                   based on feature_importances_

    Returns:
    ----------
    Print statement outputting variable importance in descending order
    """
    print("Feature Ranking : ")
    for f in range(len(names)):
        i = f
        print(
            "%d. the feature '%s' has a Mean Decrease in Gini of %f"
            % (f + 1, names[indices[i]], importance[indices[f]])
        )


def variable_importance_plot(importance, indices, names):
    """
    Purpose:
    ----------
    Prints bar chart detailing variable importance for CART model
    NOTE: feature_space list was created because the bar chart
    was transposed and index would be in incorrect order.

    importance: Array returned from feature_importances_ for CART models organized in descending order

    indices: Organized index of dataframe from largest to smallest based on feature_importances_

    Returns:
    ----------
    Returns variable importance plot in descending order
    """
    index = np.arange(len(names))
    importance_desc = sorted(importance)
    feature_space = []
    for i in range(len(names)-1, -1, -1):
        feature_space.append(names[indices[i]])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#fafafa")
    plt.title(
        "Feature importances for Random Forest Model\
    \nBreast Cancer (Diagnostic)"
    )
    plt.barh(index, importance_desc, align="center", color="#875FDB")
    plt.yticks(index, feature_space)

    plt.ylim(-1, 30)
    plt.xlim(0, max(importance_desc))
    plt.xlabel("Mean Decrease in Gini")
    plt.ylabel("Feature")

    plt.show()
    plt.close()


def cross_val_metrics(fit, training_set, class_set, print_results=True):
    """
    Purpose
    ----------
    Function helps automate cross validation processes while including
    option to print metrics or store in variable

    Parameters
    ----------
    fit: Fitted model
    training_set:  Data_frame containing 80% of original dataframe
    class_set:     data_frame containing the respective target vaues
                  for the training_set
    print_results: Boolean, if true prints the metrics, else saves metrics as
                  variables

    Returns
    ----------
    scores.mean(): Float representing cross validation score
    scores.std() / 2: Float representing the standard error (derived from cross validation score's standard deviation)
    """
    n = KFold(n_splits=10)
    scores = cross_val_score(fit, training_set, class_set, cv=n)
    if print_results:
        print(
            "Accuracy:{0:0.3f} (+/- {1:0.3f})".format(scores.mean(), scores.std()/2))
    else:
        return scores.mean(), scores.std()/2


def plot_roc_curve(fpr, tpr, auc, mod, xlim=None, ylim=None):
    """
    Purpose
    ----------
    Function creates ROC Curve for respective model given selected parameters.
    Optional x and y limits to zoom into graph

    Parameters
    ----------
    fpr:  Array returned from sklearn.metrics.roc_curve for increasing
    false positive rates
    tpr:  Array returned from sklearn.metrics.roc_curve for increasing
    true positive rates
    auc:  Float returned from sklearn.metrics.auc (Area under Curve)
    mod:  String represenation of appropriate model, can only contain the
    following: ['knn', 'rf', 'nn']
    xlim: Set upper and lower x-limits
    ylim: Set upper and lower y-limits

    Returns:
    ----------
    Returns plot of Receiving Operating Curve for specific model. Allowing user to
    specify x and y-limits.
    """
    mod_list = ['knn', 'rf', 'nn']
    method = [('Kth Nearest Neighbor', 'deeppink'),
              ('Random Forest', 'red'),
              ('Neural Network', 'purple')]

    plot_title = ''
    color_value = ''
    for i in range(0, 3):
        if mod_list[i] == mod:
            plot_title = method[i][0]
            color_value = method[i][1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#fafafa')

    plt.plot(fpr, tpr,
             color=color_value,
             linewidth=1)
    plt.title('ROC Curve For {0} (AUC = {1: 0.3f}) \
          \nBreast Cancer Diagnostic'
              .format(plot_title, auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Add Diagonal line
    plt.plot([0, 0], [1, 0], 'k--', lw=2, color='black')
    plt.plot([1, 0], [1, 1], 'k--', lw=2, color='black')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.close()
