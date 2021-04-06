import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm

# Colors you can use
colors = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange", "lightgreen"]

#######################################################################################################################
# Function plot_random():
# Function to plot random values. Task: Get familiar with plt.scatter. Plot random data. Set a title, legend, label and colors.
# https://pythonspot.com/matplotlib-scatterplot/
# Hint: Plot data for every center individually
#######################################################################################################################
def plot_random(n_samples = 500, centers = 4):
    # Creates random data (X) with labels (y)
    X, y = make_blobs(n_samples = n_samples, centers = centers, n_features = 2, random_state = 0)

    # Create plot
    for i in range(centers):
        # is our label y at position of X equal to the cluster represented by i
        plt.scatter(X[:,0][y==i], X[:,1][y==i], c=colors[i], alpha=0.5)
    
    # To eliminate guarenteed indexOutOfBounds problems
    # use zip to iterate through available colors (uncommented)
    # for i, color in zip(range(centers), colors):
    #     plt.scatter(X[:,0][y==i], X[:,1][y==i], c=color, alpha=0.5)

    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    pass

#######################################################################################################################
# Function read_wine_data(filename = 'wine.data'):
# Function to read in wine data
#
#   Tasks:
#   - Read in data as CSV (np.genfromtxt)
#   - Split data into vector of class names and remaining data
#   - Return
#
# Input arguments:
#   - [string] filename: Filename of data
# Output argument:
#   - [np] x:  data
#   - [np] y:  labels
#######################################################################################################################
def read_wine_data(filename = 'wine.data'):
    # This variable is just a summary of the column names for you.
    # cols =  ['Class', 'Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 
    #         'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', 
    #         'Hue', 'OD280/OD315', 'Proline']
    wineFile = np.genfromtxt(filename, dtype=float, delimiter=",")
    data = wineFile[:,1:]
    labels = wineFile[:,0]

    return data, labels

#######################################################################################################################
# Function calculate_pca(X, n_components = 2):  
# Function to calculate a priciple component analysis. 
# Hint: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
#
# Input arguments:
#   - [np] X: data
# Output argument:
#   - [np] X: pca
#######################################################################################################################
def calculate_pca(X, n_components = 2):    
    pca = PCA(n_components = n_components)
    # return the firstly fitted and then on X applied dimensionality reduction
    return pca.fit_transform(X)

#######################################################################################################################
# Function calculate_lda(X, y):
# Function to calculate a Linear Discriminant Analysis. 
# Hint: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
#
# Input arguments:
#   - [np] X: data
#   - [np] y: lables
# Output argument:
#   - [np] X: lda
#######################################################################################################################
def calculate_lda(X, y): 
    lda = LDA()
    # return the firstly fitted and then transformed data
    return lda.fit_transform(X, y)

#######################################################################################################################
# Function scatter_data(X, y, title = ""):
# Function to plot data per class
#
#   Tasks:
#   - Retrieve all unique class names
#   - For every class:
#       - Plot class data with color (plt.scatter)
#   - Set title
#
# Input arguments:
#   - [np] X: data
#   - [np] y: lables
#   - [String] title: ""
#######################################################################################################################
def scatter_data(X, y, title = ""):
    # unique = [int(e) for e in np.unique(y)] #creates an integer list of all centers

    # set number of classes to the right value
    unique = np.unique(y)
   
    # To eliminate guarenteed indexOutOfBounds problems
    # use zip to iterate through available colors (uncommented)
    for i, color in zip(range(len(unique)), colors):
        plt.scatter(X[:,0][y==unique[i]], X[:,1][y==unique[i]], c=color, alpha=0.5)
        
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    pass

#######################################################################################################################
# Function rescale_data(X):
# Function to scale values in range [0,1]
#
# Input arguments:
#   - [np] X: data
# Output argument:
#   - [np] X: rescaled
#######################################################################################################################
def rescale_data(X):
    # create MinMaxScaler object and set range
    scaler = MinMaxScaler(feature_range=(0,1))
    # computes the min and max of X for later transformation (fitting)
    # and then rescales (transformes) X to set range
    return scaler.fit_transform(X)

#######################################################################################################################
# Function remove_class(X, y, class_index):
# Function to remove all data entries in X and y for one class (class_index)
#
# Input arguments:
#   - [np] X: data
# Output argument:
#   - [np] X: rescaled
#######################################################################################################################
def remove_class(X, y, class_index):
    # check if integer class index is in unique centers
    assert class_index in [int(e) for e in np.unique(y)] # maybe not necessary to convert to int

    # grab all indices of elements matching the class index
    indices = [i for i, x in enumerate(y) if x == class_index]
    newY = np.delete(y, indices)
    # create "new X" matrix with new dimensions.
    newX = np.zeros(shape = (len(newY), X.shape[1]), dtype=X.dtype)
    # save the columns to the correct position in new 
    newX[:,0] = np.delete(X[:,0], indices)
    newX[:,1] = np.delete(X[:,1], indices)
    # alternative
    # newX2 = np.delete(X, indices, axis=0)

    return newX, newY

#######################################################################################################################
# Function split(X, y, test_size = 0.2):
# Function to split data into training and test set
#
# Input arguments:
#   - [np] X: data
#   - [np] y: label
#   - [Float] test_size: Percentage= test set/whole set 
# Output argument:
#   - [np] X_train, X_test, y_train, y_test
#######################################################################################################################
def split(X, y, test_size = 0.2):
    # split X and y into test and train data.
    # test_size is proportion of train to test data
    # use default random generator through prameter None
    return train_test_split(X, y, test_size = test_size, random_state=None)

#######################################################################################################################
# Function  evaluate_svm(X_train, X_test, y_train, y_test):
# Function to evaluate a SVM. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#
#   Tasks:
#   - Create SVC kernel
#   - Fit to data
#   - Print the accuracy [Hint: method 'score']
#   - Plot data points (Different class = Different color; Train<>test = Different marker)
#   - Highlight support vectors (Hint: attribute 'support_vectors_')
#   - Display decision function (use 'plot_svc_decision_function')
#   - Set title and legend for plt and show it
#
# Input arguments:
#   - [np] X_train, X_test, y_train, y_test data
#######################################################################################################################
def evaluate_svm(X_train, X_test, y_train, y_test):
    model_svc = SVC(gamma = 'auto', kernel='linear', C=1.0)
    # fit train data in SVC linear kernel
    model_svc.fit(X_train, y_train)
    print("Test data score: ")
    print(model_svc.score(X_train, y_train))
    # create dynamic class labels for legend
    labels = ["Class 1 ", "Class 2 "]
    # iterate over the classes in y_train, separate classes by color and test and train data by markers
    for i, color, label in zip(range(len(np.unique(y_train))), colors, labels):
        # scatter plot of test data
        plt.scatter(X_test[:,0][y_test==i+1], X_test[:,1][y_test==i+1], color=color, marker='o', alpha=0.9, label=label + "test data")
        # scatter plot of training data
        plt.scatter(X_train[:,0][y_train==i+1], X_train[:,1][y_train==i+1], color=color, marker='*', alpha=0.9, label=label + "training data")
    
    # call given function to generate vectors
    plot_svc_decision_function(model_svc)
    plt.legend()
    plt.show()
    pass

def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    if plot_support:
        # get support vectors and highlight with black circle
        plt.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, linewidth=1, facecolors='none', edgecolors='black')



if __name__ == '__main__':
    
    # Plot random data
    # plot_random()

    # Read in wine data
    X, y = read_wine_data()
    scatter_data(X, y, "Original data [first two features]")

    # Rescale data
    X = rescale_data(X)

    # Calculate pca
    pca_data = calculate_pca(X)
    scatter_data(pca_data, y, "PCA")

    # Calculate lda
    X = calculate_lda(X, y)
    scatter_data(X, y, "LDA")

    # Remove a class from the data
    X, y = remove_class(X, y, 3)

    # Split data into training and test set
    X_train, X_test, y_train, y_test = split(X, y, test_size=0.3)  

    # Evaluate SVM
    scatter_data(X, y, "SVM input")
    evaluate_svm(X_train, X_test, y_train, y_test)


