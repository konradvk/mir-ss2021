# import the necessary packages
import csv
import math
from pathlib import Path

#######################################################################################################################
# Function square_rooted(x)
# Function to calculate the root of the sum of all swared elements of a list
#
# Help: 
#   - for each element calculate the square value. Sum all values up. Calculate the root
# 
# Input arguments:
#   - [list] x: list of input numbers
# Output argument:
#   - [float] result
#######################################################################################################################
def square_rooted(x):
        return math.sqrt(sum([a*a for a in x]))

class Searcher:

    #######################################################################################################################
	# Function __init__(self, index_path):
	# Init function. Just set the index path
	#######################################################################################################################
    def __init__(self, index_path):
        # store our index path
        # self.index_path = "C:\Users\kvkue\Dropbox\B.Sc Med Inf\Medical Data Science\1information retrieval\index.csv"
        self.index_path = index_path

    #######################################################################################################################
	# Function search(self, queryFeatures, limit = 5)
	# Function retrieve similar images based on the queryFeatures
	#
    # 	# Tasks:
    #   - If there is no index file -> Print error and return False [Hint: Path(*String*).exists()]
    #   - Open the index file
    #   - Read in CSV file [Hint: csv.reader()]
    #   - Iterate over every row of the CSV file
    #       - Collect the features and cast to float
    #       - Calculate distance between query_features and current features list
    #       - Save the result in a dictionary: key = image_path, Item = distance
    #   - Close file
    #   - Sort the results according their distance
    #   - Return limited results
	# 
	# Input arguments:
	#   - [list] query_features: Lost of query features
    #   - [int] limit: Limit the retrieved results
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def search(self, query_features, limit = 5):
        if not Path(self.index_path).exists():
            return False

        euclidean_dict = { }
        with open(self.index_path) as csvfile:
            reader = csv.reader(csvfile) # change contents to floats
            for row in reader: # each row is a list
                tmpList = [float(i) for i in row[1:]]
                distance = self.euclidean_distance(query_features, tmpList)
                euclidean_dict[row[0]] = distance
        
        csvfile.close()
        euclidean_dict = sorted([(v, k) for (k, v) in euclidean_dict.items()])

        return euclidean_dict[:limit]

             

    
    #######################################################################################################################
	# Function euclidean_distance(self, x, y):
	# Function to calculate the euclidean distance for two lists
	#
    # 	# Help: https://pythonprogramming.net/euclidean-distance-machine-learning-tutorial/
	# 
	# Input arguments:
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def euclidean_distance(self, x, y):
        assert len(x) == len(y)
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x,y)]))

        # eucl = []
        # for i in range(0,len(x)):
        #     eucl.extend( math.sqrt((float)(x[i]*y[i])^2) )
        # return eucl


    #######################################################################################################################
	# Function manhattan_distance(self, x, y):
	# Function to calculate the manhattan distance for two lists
	# 
	# Input arguments:
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def manhattan_distance(self, x, y):
        # TODO 
        pass

    #######################################################################################################################
	# Function minkowski_distance(self, p, x, y):
	# Function to calculate the minkowski distance for two lists
	# 
    # 	# Help: We expect w to be 1
    # 
	# Input arguments:
    #   - [int] p: P-value from slide
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def minkowski_distance(self, p, x, y):
        # TODO 
        pass


    #######################################################################################################################
	# Function cosine_similarity(self, x, y):
	# Function to calculate the cosine similarity for two lists
	#
    # 	# Help:
    #       Compute numerator
    #       Compute denominator with the help of "square_rooted"
    #       Calculate similarity
    #       Change range to [0,1] rather than [-1,1]
	# 
	# Input arguments:
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed similarity
	#######################################################################################################################
    def cosine_similarity(self, x, y):
        # TODO 
        pass

    #######################################################################################################################
	# Function cosine_distance(self, x, y):
	# Function to calculate the cosine distance with help of cosine similarity
	#######################################################################################################################
    def cosine_distance(self, x, y):
        # TODO 
        pass
