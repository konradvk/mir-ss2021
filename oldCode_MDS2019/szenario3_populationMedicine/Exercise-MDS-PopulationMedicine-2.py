import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from sklearn.metrics import homogeneity_score, completeness_score 
from sklearn.metrics import silhouette_samples
from sklearn.metrics.cluster import contingency_matrix

 
#######################################################################################################################
# Function pair_plot(df):
# Function show the pair-plot of the following attributes('perimeter_mean', 'area_mean', 'smoothness_mean', 
# 'concavity_mean', 'symmetry_mean') using  seaborn for the primary analysis

# Input arguments:
#   - df : data freame of breast cancer data 
#######################################################################################################################
  
def pair_plot(df):
    # declare attributes to be plotted "against" each other.
    attributes = ['perimeter_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']
    # give the pairplots the dataframe and the columns to plot
    sns.pairplot(df, vars=attributes, hue="diagnosis", height=1.5, aspect=1.5)
    plt.show()
    
    
#######################################################################################################################
# Function project_dataset(df,cdf):
# Project the dataset on a bidimensional plane using TSNE

# Input arguments:
#   - df : data freame of breast cancer data 
#   - cdf : data freame of breast cancer data  without 'diagnosis' attribute
#######################################################################################################################

def project_dataset(df,cdf):
    
    # initiate T-SNE instance bidimensional
    tsne = TSNE(n_components=2)
    # transform the data set (without M,B diagnosis)
    data_tsne = tsne.fit_transform(cdf)

    # add the coulumns X,Y to the  to the transformed data and assign it to df_tsne  
    df_tsne = pd.DataFrame(data_tsne, columns=['x', 'y'], index=cdf.index)

    # concat the two data frames (df, df_tsne) to be able to plot data
    dff = pd.concat([df, df_tsne], axis=1)

    # print(dff)
    
    # Show the diagram
    fig, ax = plt.subplots(figsize=(18, 11))

    with sns.plotting_context("notebook", font_scale=1.5):
        sns.scatterplot(x='x',
                        y='y',
                        hue='diagnosis',
                        size='area_mean',
                        style='diagnosis',
                        sizes=(30, 400),
                        palette=sns.color_palette("husl", 2),
                        data=dff,
                        ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.show()
    return dff
    
    
    
#######################################################################################################################
# Function kmean(df,dff):
# Perform a K-Means clustering with K=2 and project the data  

# Input arguments:
#   - df : data freame of breast cancer data 
#   - dff : data freame of breast cancer data  without 'diagnosis' attribute
#######################################################################################################################

def kmean(cdf,dff, amount_cluster=2):

    #initiate Kmeans instance
    km = KMeans(n_clusters=amount_cluster, max_iter=1000, random_state=1000)

    # Apply fit_predict to cdf data frame 
    Y_pred = km.fit_predict(cdf)

    # add the coulum  'prediction' to the data frame
    df_km = pd.DataFrame(Y_pred, columns=['prediction'], index=cdf.index)

    # concat the two data frames
    kmdff = pd.concat([dff, df_km], axis=1)

    # Show the clustering result
    fig, ax = plt.subplots(figsize=(18, 11))

    with sns.plotting_context("notebook", font_scale=1.5):
        sns.scatterplot(x='x',
                        y='y',
                        hue='prediction',
                        size='area_mean',
                        style='diagnosis',
                        sizes=(30, 400),
                        # set color palette to right amount
                        palette=sns.color_palette("husl", amount_cluster),
                        data=kmdff,
                        ax=ax)
    
    # Only for information purposes:
    # clusters = km.cluster_centers_
    # scattering does not make sense here, as centroids are 30-dimensional
    # for cluster in km.cluster_centers_:
    #     plt.scatter(cluster[0], cluster[1], marker='*', s=200)
    

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.show()
  
# Compute the metrics for K=2
def compute_metrics(df,cdf,dff):
        
    
    km = KMeans(n_clusters=2, max_iter=1000, random_state=1000)
    Y_pred = km.fit_predict(cdf)
    df_km = pd.DataFrame(Y_pred, columns=['prediction'], index=cdf.index)
    kmdff = pd.concat([dff, df_km], axis=1)

    # compute the completeness and homogeneity score
     # Clustering meets homogeneity satisfaction, if all points of one cluster are memebr of the same class
    homogeneity = homogeneity_score(kmdff['diagnosis'], kmdff['prediction'])
     # Clustering meets completeness satisfaction, if all points of one class are in the same cluster
    completeness = completeness_score(kmdff['diagnosis'], kmdff['prediction'])
    # print the previously computed
    print("\n Homogeneity score: {} ".format(homogeneity))
    print("\n Completeness score: {} ".format(completeness))

    # Compute and show the contingency matrix
    cm = contingency_matrix(kmdff['diagnosis'].apply(lambda x: 0 if x == 'B' else 1), kmdff['prediction'])

    fig, ax = plt.subplots(figsize=(8, 6))

    # with sns.plotting_context("notebook", font_scale=1.5):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)

    plt.show()
    
# Perform a K-Means clustering with K=8
def kmean_8(cdf,dff):
    # Call kMeans with 8 clusters
    kmean(cdf,dff, amount_cluster=8)
   
    
if __name__ == '__main__':
        
    # For reproducibility
    np.random.seed(1000)



    bc_dataset_path = 'wdbc.data'

    bc_dataset_columns = ['id','diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
                      'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                      'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                      'radius_se','texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                      'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                      'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                      'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                      'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    
    # Load the dataset
    df = pd.read_csv(bc_dataset_path, index_col=0, names=bc_dataset_columns).fillna(0.0)
    cdf = df.drop(['diagnosis'], axis=1)
   
    # print(df)
    # pair_plot (df)
    
    dff=project_dataset(df,cdf)
    
    # default is 2 cluster
    kmean(cdf,dff)
          
    compute_metrics(df,cdf,dff)
    
    kmean_8(cdf,dff)
     
   



    