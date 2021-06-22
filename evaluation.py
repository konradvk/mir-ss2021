from preprocessing import get_images_paths
from query import Query
import cv2
import sys
import numpy as np
from pathlib import Path
import csv
import os

code_path = str(Path("static/codes/codes.csv"))

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
    if (k == None):
        k = len(correct_prediction_list)
    elif (k < 0 | k > len(correct_prediction_list)):
        print("k is greater than the length of the prediction list or negative")
        pass
    
    counter = 0

    for i in range(0, k):
        if correct_prediction_list[i]:
            counter += 1
    
    average_precision = float(counter/k)
    return average_precision


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
    precisionAtIndex = 0.0
    averagePrecision = 0.0
    for i in range(0, len(correct_prediction_list)):
        precisionAtIndex = precision_at_k(correct_prediction_list, i+1)
       
        if(correct_prediction_list[i] == True):
            averagePrecision += precisionAtIndex
    
    averagePrecision = averagePrecision/amount_relevant

    return averagePrecision



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
        amount = 0

        for row in reader:
            if row[1] == query_image_code:
                amount += 1
    f.close()
    return amount




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
    # ImageCLEFmed2007_test
    image_paths = get_images_paths(image_directory=os.path.join("static", "img_db"), file_extensions=["*.png"])

    res_dic = {}
    average = 0.0

    for i, path in enumerate(image_paths):
        if(i == limit-1):
            break

        query = Query(path)
        query_result = query.run()
        res_dic = query.check_code(query_result)

        true_list = list(res_dic.values())
        true_list.pop(0) # originalbild löschen

        amount = amount_relevant_images(query.query_image_code)

        average += average_precision(true_list, amount)
    
    return (average/limit)




if __name__ == "__main__":
    test = [True, True, False, False, True]

    print("P@K: ", precision_at_k(test))

    print("AveP: ", average_precision(test, 5))

    result = mean_average_precision(limit=10)
    print("\nMAP: ", result)