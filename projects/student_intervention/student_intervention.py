# Import libraries
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

print student_data.head()

# TODO: Calculate number of students
n_students = len(student_data)

# TODO: Calculate number of features
n_features = len(student_data.columns) - 1  # without last column

# TODO: Calculate passing students
n_passed = len(student_data[student_data['passed'] == 'yes'])

# TODO: Calculate failing students
n_failed = len(student_data[student_data['passed'] == 'no'])

# TODO: Calculate graduation rate
grad_rate = float(n_passed) / (n_passed + n_failed)

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split

# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    train_time = end - start
    # Print the results
    print "Trained model in {:.4f} seconds".format(train_time)
    return train_time


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    predict_time = end - start
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(predict_time)
    return f1_score(target.values, y_pred, pos_label='yes'), predict_time


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    train_time = train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    f1_score_train_data, train_predict_time = predict_labels(clf, X_train, y_train)
    f1_score_test_data, test_predict_time = predict_labels(clf, X_test, y_test)
    print "F1 score for training set: {:.4f}.".format(f1_score_train_data)
    print "F1 score for test set: {:.4f}.".format(f1_score_test_data)
    return train_time, f1_score_train_data, train_predict_time, f1_score_test_data, test_predict_time


# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier()
clf_C = RandomForestClassifier(n_estimators=30)

# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
tt_A1, f1_tr_A1, ct_tr_A1, f1_ts_A1, ct_ts_A1 = train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
tt_A2, f1_tr_A2, ct_tr_A2, f1_ts_A2, ct_ts_A2 = train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
tt_A3, f1_tr_A3, ct_tr_A3, f1_ts_A3, ct_ts_A3 = train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)

tt_B1, f1_tr_B1, ct_tr_B1, f1_ts_B1, ct_ts_B1 = train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
tt_B2, f1_tr_B2, ct_tr_B2, f1_ts_B2, ct_ts_B2 = train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
tt_B3, f1_tr_B3, ct_tr_B3, f1_ts_B3, ct_ts_B3 = train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)

tt_C1, f1_tr_C1, ct_tr_C1, f1_ts_C1, ct_ts_C1 = train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
tt_C2, f1_tr_C2, ct_tr_C2, f1_ts_C2, ct_ts_C2 = train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
tt_C3, f1_tr_C3, ct_tr_C3, f1_ts_C3, ct_ts_C3 = train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)

# Markdown printout
print '\n** Classifer 1 - %s**\n' % clf_A.__class__.__name__
print '| Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |'
print '| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |'
print '| %d               |  %.4f              |  %.4f                      |  %.4f                | %.4f         |' % (
len(X_train_100), tt_A1, ct_ts_A1, f1_tr_A1, f1_ts_A1)
print '| %d               |  %.4f              |  %.4f                      |  %.4f                | %.4f         |' % (
len(X_train_200), tt_A2, ct_ts_A2, f1_tr_A2, f1_ts_A2)
print '| %d               |  %.4f              |  %.4f                      |  %.4f                | %.4f         |' % (
len(X_train_300), tt_A3, ct_ts_A3, f1_tr_A3, f1_ts_A3)
print '\n** Classifer 2 - %s**\n' % clf_B.__class__.__name__
print '| Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |'
print '| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |'
print '| %d               |  %.4f              |  %.4f                      |  %.4f                | %.4f         |' % (
len(X_train_100), tt_B1, ct_ts_B1, f1_tr_B1, f1_ts_B1)
print '| %d               |  %.4f              |  %.4f                      |  %.4f                | %.4f         |' % (
len(X_train_200), tt_B2, ct_ts_B2, f1_tr_B2, f1_ts_B2)
print '| %d               |  %.4f              |  %.4f                      |  %.4f                | %.4f         |' % (
len(X_train_300), tt_B3, ct_ts_B3, f1_tr_B3, f1_ts_B3)
print '\n** Classifer 3 - %s**\n' % clf_C.__class__.__name__
print '| Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |'
print '| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |'
print '| %d               |  %.4f              |  %.4f                      |  %.4f                | %.4f         |' % (
len(X_train_100), tt_C1, ct_ts_C1, f1_tr_C1, f1_ts_C1)
print '| %d               |  %.4f              |  %.4f                      |  %.4f                | %.4f         |' % (
len(X_train_200), tt_C2, ct_ts_C2, f1_tr_C2, f1_ts_C2)
print '| %d               |  %.4f              |  %.4f                      |  %.4f                | %.4f         |' % (
len(X_train_300), tt_C3, ct_ts_C3, f1_tr_C3, f1_ts_C3)
