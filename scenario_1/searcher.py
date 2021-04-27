# import the necessary packages
import csv
import math
from pathlib import Path

from pdb import set_trace as st

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
    def search(self, queryFeatures, limit = 5):
        # initialize our dictionary of results
        results = {}

        # check if there is a csv file
        if not Path(self.index_path).exists():
            print("There is no index file!")
            return False

        # open the index file for reading
        with open(self.index_path) as f:
            # initialize the CSV reader
            reader = csv.reader(f)

            # loop over the rows in the index
            for row in reader:
                # extract features [1:] and convert to float
                features = [float(x) for x in row[1:]]

                # TODO choose a distance funtion here
                distance = self.euclidean_distance(features, queryFeatures)

                # add to dictionary; Key: image_path, Item: distance_value
                results[row[0]] = distance

            # close the reader
            f.close()

        # sort our results
        results = sorted([(v, k) for (k, v) in results.items()])
        
        # return our (limited) results
        return results[:limit]

    
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
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))       
        return distance


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
        assert len(x) == len(y)  
        distance = sum(abs(a - b) for a, b in zip(x, y))
        return distance

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
        #make sure both array are in the same dimension
        assert len(x) == len(y)  
        distance = sum([abs(a - b)^p for a, b in zip(x,y)])^1/p
        return distance


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
        assert len(x) == len(y)  
        
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = square_rooted(x)*square_rooted(y)

        # similarity in range [-1,1]
        similarity = numerator/float(denominator)

        # similarity in range [0,1]
        similarity = (similarity / 2) + 1/2
        
        return similarity

    #######################################################################################################################
	# Function cosine_distance(self, x, y):
	# Function to calculate the cosine distance with help of cosine similarity
	#######################################################################################################################
    def cosine_distance(self, x, y):
        return 1 - self.cosine_similarity(x, y)
