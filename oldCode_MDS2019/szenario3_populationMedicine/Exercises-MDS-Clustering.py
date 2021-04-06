import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from scipy.spatial import distance


#######################################################################################################################
# Function plot_random(n_samples,centers):
# Function generate and plot the data
#
# Input arguments:s
#   - int n_samples: no. of samples
#   - list centers: list of center points  hint: centers=[[-2, -1],[4,4], [1, 1]] 
# Output argument:
#   - X: Array of shape [n_samples, n_features]. (Feature Matrix)
#   - y: Array of shape [n_samples]. (Response Vector)
#######################################################################################################################
def plot_random(n_samples=2200,centers=[[-2, -1],[4,4], [1, 1]]):
    
    X, y = make_blobs(n_samples = n_samples, centers = centers, n_features = 2, random_state = 0)

    # Create plot
    plt.scatter(X[:,0], X[:,1], c=y, cmap='jet', alpha=0.8, s=2.5)

    # For custom colors uncomment following code
    # colors = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange", "lightgreen"]
    # for i in range(len(centers)):
    #     # is our label y at position of X equal to the cluster represented by i
    #     plt.scatter(X[:,0][y==i], X[:,1][y==i], c=colors[i], alpha=0.5)

    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return X,y

#######################################################################################################################
# Function Plot_elbow(clusters,X):
# Function apply the k-mean on the data and generate the plot
#
# Input arguments:
#   - int clusters: no. of clusters
#   - X: Array of shape [n_samples, n_features]. (Feature Matrix)

#######################################################################################################################


def Plot_elbow(q,X):
    # to determine the optimal k
    distortions = []
    K = range(1,q)

    # Calculate the euclidean distance and append it with distortions

    for amount_cluster in K:
        # Fit the model with data
        kmeanModel = KMeans(init = "k-means++", n_clusters=amount_cluster, n_init = 12).fit(X)
        # Get Labels and centroids
        k_means_labels = kmeanModel.labels_
        k_means_centroids = kmeanModel.cluster_centers_

        # Compute distance to assigned centroid
        # compute euclidean distance from each point to its assigned centroid
        distances = [distance.euclidean(X[idx], k_means_centroids[label]) for idx, label in enumerate(k_means_labels)]
        # Append the sum of euclidean distances divided by the amount of points
        distortions.append( sum(distances) / X.shape[0] )

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()



#######################################################################################################################
# Function K_mean_random_data(clusters,X):
# Function apply the k-mean on the data and generate the plot
#
# Input arguments:
#   - int clusters: no. of clusters
#   - X: Array of shape [n_samples, n_features]. (Feature Matrix)

#######################################################################################################################

def K_mean_random_data(clusters,X):
    
    k_means = KMeans(init = "k-means++", n_clusters = clusters, n_init = 12)
    
    #Fit the model with data
    k_means.fit(X)  
    # Assign the lables to k_means_labels
    k_means_labels = k_means.labels_
    # Assign the clutsre center to the  k_means_cluster_centers
    k_means_cluster_centers = k_means.cluster_centers_
   
    # Specify the plot with dimensions.
    fig = plt.figure(figsize=(6, 4))

    # Colors uses a color map, which will produce an array of colors based on
    # the number of labels there are. We use set(k_means_labels) to get the
    # unique labels.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

    # Create a plot
    ax = fig.add_subplot(1, 1, 1)

    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for k, col in zip(range(len(k_means_cluster_centers)), colors):

        # Create a list of all data points, where the data poitns that are 
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        my_members = (k_means_labels == k)
    
        # Define the centroid, or cluster center.
        cluster_center = k_means_cluster_centers[k]
    
        # Plots the datapoints with color col.
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
        # Plots the centroids with specified color, but with a darker outline
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

    # Set title of the plot
    ax.set_title('K-Means')
    
    #Set the x-axis label
    ax.set_xlabel('x-axis')

    #Set the x-axis label
    ax.set_ylabel('y-axis')

    # Show the plot
    plt.show()

#######################################################################################################################
# Function BSAS(data,theta,max_clusters):
# Function Implement Basic Sequential Algorithmic Scheme and plot the data
#
# Input arguments:
#   - np array of data hint.[[3 5] [1 4] [1 5]]
#   - int Theta: no. of samples
#   - int max_clusters: maximum clusters
#######################################################################################################################

def BSAS(X3, Theta=3, q=5):
    
    # number of clusters that the algorithm has already created
    m = 1
    # initial centroid
    m_cluster = [X3[0]]
    # label of data point (allocation)
    labels = [0]

    # Go through all points
    for i in range(1, X3.shape[0]):

        # Find the minimal distance to a cluster and save the index
        min_distance = distance.euclidean(X3[i], m_cluster[0])
        min_cluster_idx = 0
        for j in range(1, len(m_cluster)):
            new_dist = distance.euclidean(X3[i], m_cluster[j])
            if new_dist < min_distance:
                min_distance = new_dist
                min_cluster_idx = j
        
        # Check for Theta and max amount of cluster allowed
        if min_distance > Theta and m < q:
            # create a new cluster and add it to m_cluster
            m += 1
            m_cluster.append(X3[i])
            # index in clusters starts at 0
            labels.append(m-1)
        else:
            # update representatives...
            # calculating the mean vector
            mean_center = m_cluster[min_cluster_idx]
            sum_allies = sum([1 for i in labels if i == min_cluster_idx])
            new_ally = X3[i]
            # formula:  old center is weighted the amount of points (allies) belonging to it PLUS the new point coordinates
            # AND then normalized by all new cetner members (allies)
            m_cluster[min_cluster_idx] = [(e*sum_allies + new_ally[idx])/(sum_allies+1) for idx, e in enumerate(mean_center)]
            # assign the cluster of previously found min distance index
            labels.append(min_cluster_idx)
    

    # Plot the clustering outcome
    plt.scatter(X3[:,0], X3[:,1], c=labels, alpha=0.8)
    # Plot the mean vectors of cluster
    for center in m_cluster:
        plt.scatter(center[0], center[1], marker='*', s=100, c='black')
    
    plt.title('BSAS sequential clustering (Metric: Euclidean)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    
    
    
if __name__ == '__main__':
    
    # Plot random data
    n_samples=2000
    X,y=plot_random(n_samples)
    # Optimal no. of Cluster    
    Plot_elbow(10,X)
    #apply the k-mean on the data and plot
    clusters=3
    K_mean_random_data(clusters,X)
    
    x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
    x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
    X3 = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    BSAS(X3)
    
