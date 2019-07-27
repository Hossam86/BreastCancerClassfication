import numpy as np
import matplotlib.pyplot as plt

def print_dx_perc(dataframe, col, dx):
    """Function used to print class distribution for our data set"""
    dx_vals = dataframe[col].value_counts()
    dx_vals = dx_vals.reset_index()
    f = lambda x, y: 100 * (x / sum(y))
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

