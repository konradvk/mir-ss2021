import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# possible classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# from sklearn.model_selection import train_test_split


#from xgboost import XGBClassifier


#######################################################################################################################
# Function rescale_data(X):
# Function to scale values in range [0,1]
#######################################################################################################################
def min_max_normalization(X):
    X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_scaled

#######################################################################################################################
# Function split_into_training_testing(X, y, test_size = 0.2):
# Function to split data into training and test set. Function was introduced last session.
#######################################################################################################################
def split_into_training_testing(X, y, test_size = 0.2):
    if X is None or y is None:
        return None, None, None, None

    num_elements = X.shape[0]

    indices = np.ones(num_elements)
    indices[:int(num_elements*test_size)] = 0
    np.random.shuffle(indices)
    
    indices = indices.astype(bool)

    X_train = X[indices]
    y_train = y[indices]

    X_test = X[indices == False]
    y_test = y[indices == False]

    return X_train, X_test, y_train, y_test

#######################################################################################################################
# Function remove_id_gender(data):
# Function to remove the first two columns (id, gender)
#######################################################################################################################
def remove_id_gender(data):
    if data is None:
        return None
    return data[:,2:]

#######################################################################################################################
# Function accuracy(actual, predicted):
# Function to calculate accuracy. Acc([1,2,3], [1,1,1]) = 0.33
#
# Input arguments:
#   - [list] actual
#   - [list] predicted
# Output argument:
#   - [float] accuracy
#######################################################################################################################
def accuracy(actual, predicted):
    assert len(actual) == len(predicted)
    # add 1 to eq_elements if elements in two lists at same index position are equal
    eq_elements = sum([1 for i, j in zip(actual, predicted) if i == j])
    # works as a float division in Python 3
    return eq_elements/len(actual)

#######################################################################################################################
# Function sensitivity(actual, predicted, positive = 1):
# Function to calculate accuracy. sensitivity([0,0,1], [1,0,1]) = 1.0
#
#   Hint: Use relation_true_to_all(actual, predicted, value)
# 
# Input arguments:
#   - [list] actual
#   - [list] predicted
# Output argument:
#   - [float] sensitivity
#######################################################################################################################
def sensitivity(actual, predicted, positive = 1):
    return relation_true_to_all(actual, predicted, positive)

#######################################################################################################################
# Function (actual, predicted, negative = 0):
# Function to calculate accuracy. sensitivity([0,0,1], [1,0,1]) = 0.5
#
#   Hint: Use relation_true_to_all(actual, predicted, value)
#
# Input arguments:
#   - [list] actual
#   - [list] predicted
# Output argument:
#   - [float] specificity
#######################################################################################################################
def specificity(actual, predicted, negative = 0):
    return relation_true_to_all(actual, predicted, negative)

#######################################################################################################################
# Function relation_true_to_all(actual, predicted, value):
# Function to calculate the relation between the correctly predicted elements for value and the actual amount.
#
#   Tasks:
#   - Check if len of actual and predicted is the same
#   - Save the amount of elements equal to value in actual
#   - Save the amount of correctly predicted elements equal to value (same position in actual/predicted + element is value)
#   - return the ratio
#
#
# Input arguments:
#   - [list] actual
#   - [list] predicted
# Output argument:
#   - [float] specificity
#######################################################################################################################
def relation_true_to_all(actual, predicted, value):
    assert len(actual) == len(predicted)
    # following variables represent the amount of values that are equal to given 'value'
    amount_actual = sum([1 for i in actual if i == value])
    # amount_correctly_predicted = sum([1 for a,p in zip(actual,predicted) if (p == value and a == p) ])
    # if value=1 amount_false_positives, else amount_false_negatives
    amount_FP_or_FN = sum([1 for a,p in zip(actual,predicted) if ((p != value) and (a != p)) ])
    # Do not divide by zero
    try:
        ratio = amount_actual / (amount_actual + amount_FP_or_FN)
        return ratio
    except:
        # Division by zero caught return 0 as no actuals are present
        return 0.0

        

#######################################################################################################################
# Function confusion_matrix(actual, predicted):
# Function to calculate a confusion matrix. confusion_matrix([1,2,3], [1,1,1]) = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
#
#   Tasks:
#   - Assert actual and predicted have same length
#   - Create a list ("all_elements") with all elements of actual and predicted
#   - Save unique elements of "all_elements" in "unique"
#   - Save the number of different classes in "num_classes"
#   - Create an empty confusion matrix (filled with 0). Shape is "num_classes" x "num_classes"
#   - Create a "lookup" dict: Classname = index in confusion matrix
#   - Increment values of confusion matrix according to actual and predicted
#   - Return unique and matrix
#
# Input arguments:
#   - [list] actual
#   - [list] predicted
#   - [String] patient_id
# Output argument:
#   - [list] unique
#   - [list of list] matrix: confusion matrix
#######################################################################################################################
def confusion_matrix(actual, predicted):
    assert len(actual) == len(predicted)
    all_elements = np.concatenate((actual, predicted), axis=None)
    unique = np.unique(all_elements)
    num_classes = len(unique)
    # declare empty confusion amtrix with zeros
    conf_matrix = np.zeros((num_classes, num_classes), dtype=float)
    # e is key (e.g. healthy or sick) and i is the index in the matrix
    dict_tuples = [(e,i) for i,e in enumerate(unique)]
    dict_lookup = dict(dict_tuples)

    # Increment values of confusion matrix according to actual and predicted
    for idx in range(len(predicted)):
        # predicted is the x axis, should go from 0 to num_classes (left->right)
        x = dict_lookup[predicted[idx]]
        # actual is the y axis, should go from 0 to num_classes (top->down)
        y = dict_lookup[actual[idx]]
        # increment the value according to actual and predicted location
        conf_matrix[y][x] += 1
    
    return unique, conf_matrix

#######################################################################################################################
# Function visualize_confusion_matrix(unique, matrix, title = "Confusion matrix"):
# Function to visualize a confusion matrix. 
#
#   Tasks:
#   - "ax.set": Set xticks, yticks, xticklabels, yticklabels, title, ylabel, xlabel
#   -  Iterate over every element in matrix
#       - Write text "ax.text"
#
# Input arguments:
#   - [list] unique
#   - [list of list] matrix
#   - [String] title = "Confusion matrix"
#######################################################################################################################
def visualize_confusion_matrix(unique, matrix, title = "Confusion matrix"):
    if (unique is None or matrix is None):
        return None
    plt.close()
    cm = np.array(matrix)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap= "Blues")
    ax.figure.colorbar(im, ax=ax)
    # x, y represent the uniquely existing labels
    ax.set(xticks=np.arange(len(unique)), yticks=np.arange(len(unique)),
            xticklabels=unique, yticklabels=unique,
            title=title, xlabel="predicted labels", ylabel="actual labels")
 
    # Write the amount of matching "hits" to every cell of the confusion matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ax.text(i,j, int(matrix[j][i]))
    
    # set the ylim to show correct squares
    plt.ylim( top=-0.5, bottom=len(unique)-0.5 ) 
    plt.show()
    pass


#######################################################################################################################
# Function is_patient_id_invalid(patient_id)
# Function to check if patient_id (INT) is valid (>= 0 and <252)
#######################################################################################################################
def is_patient_id_invalid(patient_id):
    return (patient_id >= 0 and patient_id < 252)

#######################################################################################################################
# Function read_in_data(feature_file = "pd_speech_features.csv"):
# Function to read in data from CSV file "pd_speech_features.csv". User id in first column, class in last column
# Returns data and label
#######################################################################################################################
def read_in_data(feature_file = "pd_speech_features.csv"):
    speech_features = np.genfromtxt(feature_file, dtype=float, delimiter=",")
    # first row is feature parent category, second row is name of columns -> not needed for data
    last_row = len(speech_features[0])-1
    # print("Sick (label=1): ", sum([1 for i in speech_features[2:, last_row] if i == 1]))
    # print("Healthy (label=0): ", sum([1 for i in speech_features[2:, last_row] if i == 0]))

    data = speech_features[2:,:last_row-1] # do not include last column, as it is the label
    labels = speech_features[2:,last_row]
    return data, labels

#######################################################################################################################
# Function split_into_training_testing_with_indices(data, label, testing_indices):
# Function to split dataset into test and training set. Elements with index in "testing_indices" are part of the test set.
#
#
# Input arguments:
#   - [np] data
#   - [np] label
#   - [list] testing_indices
# Output argument:
#   - [np] training_data, training_label, testing_data, testing_label
#######################################################################################################################
def split_into_training_testing_with_indices(data, label, testing_indices):
    # testing_indices is a list of integer indices
    assert max(testing_indices) < len(label)

    # Normalizing data for test purposes of classification performance
    data_without_id_and_gender = remove_id_gender(data)
    normalized_data = min_max_normalization(data_without_id_and_gender)

    # "uncomment" if normalized data is NOT wanted
    # normalized_data = data

    testing_data = normalized_data[testing_indices]
    testing_label = label[testing_indices]
    # alternative...
    # testing_data = [data[i] for i in testing_indices]
    # testing_label = [label[i] for i in testing_indices]

    # delete does not modify the data input, even though that would also
    # be ok for this application
    training_data = np.delete(normalized_data, testing_indices, axis=0)
    training_label = np.delete(label, testing_indices)

    ########################################################
    # testing_indices is a list of booleans
    # training_data = data[testing_indices == False]
    # training_label = label[testing_indices == False]

    # testing_data = data[testing_indices]
    # testing_label = label[testing_indices]

    return training_data, testing_data, training_label, testing_label

#######################################################################################################################
# Function leave_one_subject_out(data, label, patient_id):
# Function to split dataset into test and training set. Elements of patient (patient_id) are part of the test set.
# Rest is part of training set.
#
#   Hint: Use "split_into_training_testing_with_indices"
#
# Input arguments:
#   - [np] data
#   - [np] label
#   - [String] patient_id
# Output argument:
#   - [np] training_data, training_label, testing_data, testing_labe
#######################################################################################################################
def leave_one_subject_out(data, label, patient_id):
    assert is_patient_id_invalid(patient_id)
    # Grab all indices matching to the given patient id
    indices_patient = [i for i,e in enumerate(data[:,0]) if e==patient_id]
    return split_into_training_testing_with_indices(data, label, indices_patient)

#######################################################################################################################
# Function train_evaluate_clf(clf, X_train, X_test, y_train, y_test, display_confusion_matrix = True):
# Function to fit the classifier with training data and predict the outcome on test data.
# Return the accuracy, sensivity and specifity in a list
#
# Input arguments:
#   - [np] X_train, X_test, y_train, y_test
#   - [bool] display_confusion_matrix = False: Boolean whether to display a confusion matrix or not
# Output argument:
#   - [np] acc
#######################################################################################################################
def train_evaluate_clf(clf, X_train, X_test, y_train, y_test, display_confusion_matrix = False):
    # fit the classifier to training data
    clf.fit(X_train,y_train)
    # calculate the prediction for our test data
    y_predicted = clf.predict(X_test)
    # calculate the statistics of the actual values and the prediction of the clf
    statistics = [accuracy(y_test, y_predicted), sensitivity(y_test, y_predicted), specificity(y_test, y_predicted)]

    # display confusionMatrix of current evaluation
    if display_confusion_matrix:
        unique, cm = confusion_matrix(y_test, y_predicted)
        visualize_confusion_matrix( unique, cm )
    
    return statistics


#######################################################################################################################
# Function cross_subject_validation(data, label, clf):
# Function to validate a classifier using leave one subject out validation
#
#   Task:
#   - Iterate over every patient in X
#       - Create a training and testing set for this user (leave one subject out)
#       - Run train evaluation method
#       - Save the score
#   - Return average of all scores
#
# Input arguments:
#   - [np] data
#   - [np] label
# Output argument:
#   - [float] average score
#######################################################################################################################
def cross_subject_validation(clf, data, label):
    # get all different patients (IDs)
    unique = np.unique(data[:,0])
    sum_scores = np.zeros(3)
    for patient_id in unique:
        # split data into patient and non patient part
        training_data, testing_data, training_label, testing_label = leave_one_subject_out(data, label, patient_id)
        # calculate current score of left out subject
        subject_score = train_evaluate_clf(clf,training_data, testing_data, training_label, testing_label)
        # sum up each variable in list element-wise with new list of values
        sum_scores = np.add(sum_scores, subject_score)
    
    # divide the summed variables by anount of patients
    return [x/len(unique) for x in sum_scores]

#######################################################################################################################
# Function calculate_feature_importance(X,y):
# Function to calculate the importance of features
#######################################################################################################################
def calculate_feature_importance(X,y):
    pass


#######################################################################################################################
# Function keep_most_important_features(X,y):
# Function to keep the most important features
#######################################################################################################################
def keep_most_important_features(data, label, amount = 20):
    pass

#######################################################################################################################
# Main
#######################################################################################################################
if __name__ == "__main__":

    # Static declaration of actual and predicted for testing
    actual =    list( np.arange(4) )
    actual =    [1,2,3]
    actual =    [1,2,3,4,5,1,1,3]

    predicted = list( np.arange(4) )
    predicted = [1,1,1]
    predicted =    [1,1,1,4,5,1,5,3]

    # Calculate accuracy
    acc = accuracy(actual, predicted)
    print("Accuracy for example " , actual , ", " , predicted , ": ", acc)

    # relation_true_to_all(actual, predicted, 3)

    # Visualize a confusion matrix
    unique, conf_matrix = confusion_matrix(actual, predicted)
    # visualize_confusion_matrix(unique, conf_matrix)

    # Read in the parcinson data
    data, label = read_in_data()
    # leave_one_subject_out(data, label, patient_id = 29)
    data_without_id = remove_id_gender(data)


    # Split dataset
    # X_train, X_test, y_train, y_test = split_into_training_testing_with_indices(data_without_id, label, [2,3,4,6])  
    X_train, X_test, y_train, y_test = split_into_training_testing(data_without_id, label, test_size = 0.2)  

    # Test decision tree on whole dataset
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    result = train_evaluate_clf(clf, X_train, X_test, y_train, y_test, display_confusion_matrix = True)
    print("Evaluation classifier (test_size 0.2) : {} [acc, sensivity, specifity]".format(result))

    # Validate classifier with leave_one_subject_out
    classifiers = [
    KNeighborsClassifier(5),
    # KMeans(2),
    # SVC(kernel="linear", C=0.025, probability=True, gamma='auto'),
    # NuSVC(probability=True, gamma='auto'),
    # DecisionTreeClassifier(),
    # RandomForestClassifier(n_estimators=50),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(),
    # GaussianNB(),
    # LinearDiscriminantAnalysis(),
    # QuadraticDiscriminantAnalysis()
    ]

    # Loop to test all classifiers
    for clf in classifiers:
        print("="*30)
        print("Classifier: ", clf.__class__.__name__)
        print('****Results****')
        result = cross_subject_validation(clf, data, label)
        print("\n Cross subject validation: {} [acc, sensivity, specifity]".format(result))

    print("="*30)


