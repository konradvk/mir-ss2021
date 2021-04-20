from preprocessing import get_images_paths
from query import Query
import cv2
import sys
import numpy as np
from pathlib import Path
import csv
import os


code_path = "codes.csv"

#######################################################################################################################
# Function precision_at_k(correct_prediction_list, k = None)
# Function to calculate the precision@k for this query
#
#   Tasks:
#   - If k is not defined -> k should be length of list
#   - If k > length -> Error
#   - If k < length -> cut off correct_prediction_list at k
#   - Calculate precision for list
#
# 
# Input arguments:
#   - [dict] correct_prediction_list: list with the True/False for the retrieved images
#   - [int] k
# Output argument:
#   - [float] average precision
#######################################################################################################################
def precision_at_k(correct_prediction_list, k = None):
    count = 0.0
    length = len(correct_prediction_list)

    if k is None:
        k = len(correct_prediction_list)
    elif k < 0 | k > length:
        print("k is greater than the length of the prediction list")
        return False

    for i,e in enumerate(correct_prediction_list):
        if e:
            count += 1
        if(i == k):
            break
    return count/len(correct_prediction_list)




#######################################################################################################################
# Function average_precision(self, amount_relevant):
# Function to calculate the average precision for correct_prediction_list
# 
#   Task:
#   - Calculate precision for list
#
#
# Input arguments:
#   - [dict] correct_prediction_list: list with the True/False for the retrieved images
#   - [int] amount_relevant: # relevant documents for the query
# Output argument:
#   - [float] average precision
#######################################################################################################################
def average_precision(correct_prediction_list, amount_relevant):
    sum_precision = 0
    count = 0
    for i,e in enumerate(correct_prediction_list):
        if e:
            count += 1
            sum_precision += count/(i+1)
    return sum_precision / amount_relevant


#######################################################################################################################
# Function amount_relevant_images(self, image_name): 
# Function to retrieve the amount of relavant images for a image name
# 
#   Tasks:
#   - Check if path to "code_path" exists: if not -> print error and return False
#   - Iterate over every row of code file
#   - Count the amount of codes queal to "query_image_code"
#
# Input arguments:
#   - [String] image_name: Name of the image
# Output argument:
#   - [int] amount
#######################################################################################################################
def amount_relevant_images(query_image_code): 
    if not Path(code_path).exists():
        print("The file in code_path does not exist")
        return False

    with open(code_path) as f:
        reader = csv.reader(f)
        matches = 0
        for row in reader:
           if row[1] == query_image_code:
            matches += 1
    f.close()
    return matches - 1



#######################################################################################################################
# Function mean_average_precision():
# Function to calcualte the mean average precision
# 
#   Tasks:
#   - Iterate over every image path
#      - Create and run a query for each image
#      - Compute correct_prediction_dictionary
#      - Create a list from the dict
#      - Remove the first element (its the query image)
#      - Compute amount of relevant images (function)
#      - Compute AP (function) and save the value
#   - Compute mean of APs
#
# Input arguments:
# Output argument:
#   - [float] mean average precision
#######################################################################################################################
def mean_average_precision(limit = 20):
    # get image paths of all  images
    image_paths = get_images_paths(image_directory = "C:/Users/kvkue/Pictures/MDStestdata/ImageCLEFmed2007_test/*", file_extensions = (".png"))
    # retrieve amount of images
    # average_precision(correct_prediction_list, amount_relevant)
    # amount_relevant_images(imageCode)
    av_precision = 0.0

    for i,path in enumerate(image_paths):
        if(i == limit):
            break
        
        # create object with image path and compute distance
        # as well as correct_predition_dictionary
        query = Query(path)
        distance_result = query.run()
        correct_pred_dict = query.check_code(distance_result)
        # cast dict to boolean list
        listOfBool =  list(correct_pred_dict.values())
        listOfBool.remove(0)
        # retrieve the actual amount of relevant images for the query image
        # the query image code was set during call of check_code in Query
        amount_rel = amount_relevant_images(query.query_image_code)
        # add average precision of current query image
        av_precision += average_precision(listOfBool,amount_rel)        
    
    return av_precision/limit


if __name__ == "__main__":
    test = [True, True, False, False]
    print("P@K: ", precision_at_k(test))

    print("AveP: ", average_precision(test, 5))

    # print(amount_relevant_images('3145'))
    print(amount_relevant_images('1123-127-500-000'))
    
    result = mean_average_precision(limit = 10)
    print("\nMAP: ", result)