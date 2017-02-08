# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
# import visuals as vs

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from scipy.stats.mstats import normaltest


def bcox_transform(dataframe):
    from scipy.stats import boxcox
    new_df = dataframe.copy(deep=True)
    for cc in dataframe.columns.values:
        v = dataframe[cc].values
        bc, _ = boxcox(v)
        new_df[cc] = bc
    return new_df


# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis=1, inplace=True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except IOError:
    print "Dataset could not be loaded. Is the dataset missing?"


# train test split
data_train, data_test = train_test_split(data, test_size=0.25, random_state=0)

scores = dict()

for column in data.columns.values:
    X_train, X_test = data_train.drop([column], axis=1), data_test.drop([column], axis=1)
    y_train, y_test = data_train[column], data_test[column]
    dtr = DecisionTreeRegressor(random_state=0)
    dtr.fit(X_train, y_train)
    scores[column] = dtr.score(X_test, y_test)
print('| Feature | R^2 |')
print('| ------- |:------:|')
for w in sorted(scores, key=scores.get, reverse=True):
    print('|%s|%.4f|' % (w, scores[w]))


print('Normality tests')
print('Original data')
k2, pval = normaltest(data, axis=0)
print('k2')
print(k2)
print('pval')
print(pval)
print('Log data')
k2, pval = normaltest(np.log(data), axis=0)
print('k2')
print(k2)
print('pval')
print(pval)
print('Boxcox data')
k2, pval = normaltest(bcox_transform(data), axis=0)
print('k2')
print(k2)
print('pval')
print(pval)





print('data means:\n', data.mean())
print('data median:\n', data.median())
#print('R^2 score of predicting %s: %f' % (column, r2))





