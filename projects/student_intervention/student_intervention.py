"""
Script for testing different classifiers on the data set.
    Gaussian Naive Bayes (GaussianNB)
    Decision Trees
    Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
    K-Nearest Neighbors (KNeighbors)
    Stochastic Gradient Descent (SGDC)
    Support Vector Machines (SVM)
    Logistic Regression
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_features, train_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score

# Read student data
student_data = pd.read_csv('student-data.csv')
n_students = len(student_data)
n_features = len(student_data.columns) - 1
n_passed = len(student_data[student_data['passed'] == 'yes'])
n_failed = len(student_data[student_data['passed'] == 'no'])
grad_rate = float(n_passed) / (n_passed + n_failed)

# Print the results
print("Total number of students: {}".format(n_students))
print("Number of features: {}".format(n_features))
print("Number of students who passed: {}".format(n_passed))
print("Number of students who failed: {}".format(n_failed))
print("Graduation rate of the class: {:.2f}%".format(grad_rate))

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the list of columns
print("Feature columns:\n{}".format(feature_cols))
print("Target column: {}".format(target_col))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

# do the split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=42)

print('\n\n********** classifiers **************\n')

# Naive Bayes, no extra parameters
clf = GaussianNB()
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('Gaussian NB accuracy: %.4f' % acc_score)

# Decision tree
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('default decision tree accuracy: %.4f' % acc_score)

# parameter search for decision tree
dt = DecisionTreeClassifier()
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": range(2, 100, 2)}

clf = GridSearchCV(dt, param_grid)
clf.fit(X_train, y_train)
bp = clf.best_params_
print('estimated parameters')
for k, v in bp.items():
    print("\t{:<20s}: {}".format(k, v))

clf = DecisionTreeClassifier(**bp, random_state=0)
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('Decision tree accuracy: %.4f' % acc_score)
