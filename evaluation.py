from preprocessing import get_images_paths
from query import Query
import numpy as np
from pathlib import Path
import csv
import os
from tqdm import tqdm

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
    
    counter = 0.0

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
def average_precision(correct_prediction_list, amount_relevant=None):
    if (amount_relevant == None):
        amount_relevant = len(correct_prediction_list)
    
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
# def amount_relevant_images(query_image_code): 

#     if not Path(code_path).exists(): 
#         print("The file in code_path does not exist")
#         return False

#     with open(code_path) as f:
#         reader = csv.reader(f)
#         amount = 0

#         for row in reader:
#             if row[1] == query_image_code:
#                 amount += 1
#     f.close()
#     return amount


def irma_frequency_dict():
    '''
        Method to retrieve a dictionary with the amount of an individual IRMA code.
    -----
    Returns
        Dictionary of irma code with their corresponding frequencies
    '''
    if not Path(code_path).exists(): 
        print("The file in code_path does not exist")
        return False

    # Write all irma codes into a list
    irm_code_list = []
    with open(code_path) as f:
        reader = csv.reader(f)
        for row in reader:
            irm_code_list.append(row[1])
    f.close()

    # Go through list, set to one if item not in dictionary yet. Increment if entry already exists
    dict_irma_frequency = {}
    for item in irm_code_list:
        if (item in dict_irma_frequency):
            dict_irma_frequency[item] += 1
        else:
            dict_irma_frequency[item] = 1

    return dict_irma_frequency




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
def mean_average_precision(limit = 10000):

    # get image paths of all  images
    # ImageCLEFmed2007_test
    image_paths = get_images_paths(image_directory=os.path.join("static", "img_db"), file_extensions=["*.png"])

    irma_frequencies = irma_frequency_dict()
    
    # create query object with "decoy" first image path
    query = Query(image_paths[0])
    # Make dict to receive code for image names
    # check if there is a csv file
    if not Path(query.code_path).exists():
        print("There is no code file: ", query.code_path)
        return {}

    codes = {}
    # open the code file for reading
    with open(query.code_path) as f:
        # initialize the CSV reader
        reader = csv.reader(f)

        # loop over the rows in the index
        for row in reader:
            # add to dictionary; Key: file name, Item: IRMA code
            codes[row[0]] = row[1]

    average = []
    for path in tqdm(image_paths, desc='MAP calc: '):

        # Following three lines could be time optimized...
        # A query object is created for each path and in check_code the codes.csv is read in
        query.set_image_name(path)
        pure_image_name = path.split(os.sep)[-1][:-4]
        query_result = query.run(limit=limit)

        # res_dic = query.check_code(query_result)
        # true_list = list(res_dic.values())
        true_list = [codes[pure_image_name]==codes[code.split(os.sep)[-1][:-4]] for (_, code) in query_result]

        true_list.pop(0) # originalbild löschen

        amount = irma_frequencies[codes[pure_image_name]]

        average.append(average_precision(true_list, amount))
    
    return np.mean(average)




if __name__ == "__main__":
    test = [True, True, False, True]

    print("P@K: ", precision_at_k(test))

    print("AveP: ", average_precision(test, 3))

    result = mean_average_precision(limit=10)
    # result = mean_average_precision()
    print("\nMAP: ", result)