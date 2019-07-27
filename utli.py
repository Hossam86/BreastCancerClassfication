def print_dx_perc(dataframe, col,dx):
    """Function used to print class distribution for our data set"""
    dx_vals = dataframe[col].value_counts()
    dx_vals = dx_vals.reset_index()
    f = lambda x, y: 100 * (x / sum(y))
    for i in range(0,len(dx)):
        print('{0} accounts for {1:.2f}% of the diagnosis class'.format(dx[i],f(dx_vals[col].iloc[i],dx_vals[col])))

