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
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
print('Gaussian NB accuracy: %.4f\n' % acc_score)

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
print('Decision tree accuracy: %.4f\n' % acc_score)

# Bagging with decision tree
dt = DecisionTreeClassifier(random_state=0)
clf = BaggingClassifier(base_estimator=dt, n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('default Bagging with decision tree accuracy: %.4f\n' % acc_score)

# Bagging with decision tree
gnb = GaussianNB()
clf = BaggingClassifier(base_estimator=gnb, n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('default Bagging with gnb: %.4f\n' % acc_score)

# random forest
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('default random forest: %.4f\n' % acc_score)

rf = RandomForestClassifier(random_state=0)
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": range(2, 100, 2)}
clf = GridSearchCV(rf, param_grid)
clf.fit(X_train, y_train)
bp = clf.best_params_
print('estimated parameters')
for k, v in bp.items():
    print("\t{:<20s}: {}".format(k, v))

clf = RandomForestClassifier(**bp, random_state=0)
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('Random forest accuracy: %.4f\n' % acc_score)

# random forest
clf = AdaBoostClassifier(random_state=0)
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('default AdaBoost: %.4f\n' % acc_score)

# KNN
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('default knn: %.4f\n' % acc_score)

# SGD
clf = SGDClassifier()
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('default sgdc: %.4f\n' % acc_score)

# SVM
clf = SVC()
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('default svm: %.4f\n' % acc_score)

svm = SVC()
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
clf = GridSearchCV(svm, param_grid)
clf.fit(X_train, y_train)
bp = clf.best_params_
print('estimated parameters')
for k, v in bp.items():
    print("\t{:<20s}: {}".format(k, v))

clf = SVC(**bp)
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('SVM accuracy: %.4f\n' % acc_score)


# Logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_p = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_p)
print('default logistic regression: %.4f\n' % acc_score)


