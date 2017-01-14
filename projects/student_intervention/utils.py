import pandas as pd
from time import time
from sklearn.metrics import f1_score


def preprocess_features(features):
    """ Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. """

    # Initialize new output DataFrame
    output = pd.DataFrame(index=features.index)

    # Investigate each feature column for the data
    for col, col_data in features.items():

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


def train_classifier(clf, x_train, y_train):
    """ Fits a classifier to the training data. """

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(x_train, y_train)
    end = time()
    train_time = end - start
    # Print the results
    print("Trained model in {:.4f} seconds".format(train_time))
    return train_time


def predict_labels(clf, features, target):
    """ Makes predictions using a fit classifier based on F1 score. """

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    predict_time = end - start
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(predict_time))
    return f1_score(target.values, y_pred, pos_label='yes'), predict_time


def train_predict(clf, x_train, y_train, x_test, y_test):
    """ Train and predict using a classifier based on F1 score. """

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(x_train)))

    # Train the classifier
    train_time = train_classifier(clf, x_train, y_train)

    # Print the results of prediction for both training and testing
    f1_score_train_data, train_predict_time = predict_labels(clf, x_train, y_train)
    f1_score_test_data, test_predict_time = predict_labels(clf, x_test, y_test)
    print("F1 score for training set: {:.4f}.".format(f1_score_train_data))
    print("F1 score for test set: {:.4f}.".format(f1_score_test_data))
    return train_time, f1_score_train_data, train_predict_time, f1_score_test_data, test_predict_time

