# pylint: disable=no-member
import numpy as np
import cv2
from feature_extractor import FeatureExtractor
from searcher import Searcher
import easygui
from pathlib import Path
import csv
from pdb import set_trace as st

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grand_parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, grand_parent_dir)
import get_path

#######################################################################################################################
# Function get_filename_from_path(self, path_to_file, file_ending = '.png'):
# Function to retrieve a file name from a file path
#
# Hint: "./images/739609.png" -> "739609". 
# You can use 'os.path.basename'
# 
# Input arguments:
#   - [string] path_to_file: Path to a file.
#   - [string] file_ending: String of the file type. Default = '.png'
# Output argument:
#   - [string] name: Name of the file
######################################################################################################################
def get_filename_from_path(path_to_file, file_ending = '.png'):
    filename = os.path.basename(path_to_file)

    if filename.endswith(file_ending):
        filename = filename.replace(file_ending,'')

    return filename

class Query:
    # TODO change:
    output_name = "index.csv"
    code_path = "codes.csv"
    image_directory = get_path.ImageCLEFmed2007()
    
    #######################################################################################################################
	# Function __init__(self, query_image_name = None):
	# Init function. Just call set_image_name
	#######################################################################################################################
    def __init__(self, query_image_name = None, limit = 10):
        self.limit = limit
        self.image_features = None
        self.set_image_name(query_image_name)
        self.query_image_code = None

    #######################################################################################################################
	# Function set_image_name(self, query_image_name = "./images/739562.png"):
	# Function to set the image name. Afterwards the image is loaded
	#
	# Steps: Set variable "self.query_image_name". Load image and set to "self.query_image". Call "calculate_features(self)"
    # Hint: You can also use a file dialogue to automatic enter the filename 
    #       -> http://easygui.sourceforge.net/api.html?highlight=fileopenbox#easygui.fileopenbox 
	# 
	# Input arguments:
	#   - [string] query_image_name: Path to image name. Default: like "./images/739609.png"
	#######################################################################################################################
    def set_image_name(self, query_image_name = None):
        if query_image_name is None:
            query_image_name = easygui.fileopenbox(default = self.image_directory, filetypes = ["*.png"])
        self.query_image_name = query_image_name
        self.query_image = cv2.imread(query_image_name, cv2.IMREAD_GRAYSCALE)
        self.calculate_features()

     #######################################################################################################################
	# Function calculate_features(self):
	# Function to calculate featres
	#
	# Steps: Check if "self.query_image" is None -> exit(). Extract features wit "FeatureExtractor" and set to "self.features"
	#######################################################################################################################
    def calculate_features(self):
        if self.query_image is None:
            print("There is no image named: ", self.query_image_name)
            exit()
        # initialize the FeatureExtractor
        feature_extractor = FeatureExtractor()
        # describe the query_image
        self.features = feature_extractor.extract(self.query_image)

    #######################################################################################################################
	# Function run(self):
	# Function to start a query
	#
	# Steps: 
    #   Check if "self.query_image" or self.features is None -> return. 
    #   Create a Searcher and run a search
	#######################################################################################################################
    def run(self):
        # perform the search
        searcher = Searcher(self.output_name)
        results = searcher.search(self.features, self.limit)

        # If we do not get any results, we quit
        if (results is False):
            quit()
        return results

    

    #######################################################################################################################
	# Function check_code(self, query_result):
	# Function to check if the codes of the retrieved images are equal to the code of the query_image
	#
    # Steps:
    #   - Read in the csv file with the codes
    #   - Create a Dictionary 'codes' similar to the csv file -> key : file name , item : IRMA code
    #   - Creat a Dictionary 'coorect_prediciton' with all retrieved images -> key : file name, item : boolen if the same to query_image
	# 
	# Input arguments:
	#   - [list] query_result: Result of 'run'
    # Output argument:
    #   - [Dictionary] correct_prediction:  key : file name, item : boolen
	#######################################################################################################################
    def check_code(self, query_result):
        # check if there is a csv file
        if not Path(self.code_path).exists():
            print("There is no code file: ", self.code_path)
            return {}

        codes = {}

        # open the code file for reading
        with open(self.code_path) as f:
            # initialize the CSV reader
            reader = csv.reader(f)

            # loop over the rows in the index
            for row in reader:
                # add to dictionary; Key: file name, Item: IRMA code
                codes[row[0]] = row[1]

        # get the name 
        query_image_name = get_filename_from_path(self.query_image_name)

        # If we cannot find the code for the query image
        if query_image_name not in codes:
            print("There is no code for: ", query_image_name)
            # Return empty dictionary
            return {}

        # get the code of the query image
        self.query_image_code = codes[query_image_name]

        correct_prediction = {}

        # loop over each retrieved element
        for result_image in query_result:
            # get path to file
            path_to_image = result_image[1]
            # get name of file
            image_name = get_filename_from_path(path_to_image)

            # check if there is a code for the image
            if image_name in codes:
                # get code of file
                image_code = codes[image_name]
                # save in dictionary if it was a correct prediction
                correct_prediction[image_name] = (image_code == self.query_image_code)
            else:
                print("There is no code for: ", image_name)

        return correct_prediction
            

    #######################################################################################################################
	# Function visualize_result(self, query_result, correct_prediction_dictionary):
	# Function tovisualize the results of the previous functions
	#
    # Steps:
    #   - Read in and resize (200, 200) the query image (color)
    #   - Loop over the retrieved results:
    #       - Read in the retrieved image (color)
    #       [- Retrieve whether the code is similar to query_image]
    #       [- Add a border depending on the code around the image (cv2.copyMakeBorder)]
    #       - Resize the image (200, 200)
    #       - Concatenate it to the query_image
    #   - Display the result
    #   - WaitKey
    #   - destroyWindow
	# 
	# Input arguments:
	#   - [list] query_result: Result of 'run'
    #   [- [Dictionary] correct_prediction:  Results of check_code]
	#######################################################################################################################
    def visualize_result(self, query_result, correct_prediction_dictionary, image_size = (100,100)):
        
        query_image = cv2.imread(self.query_image_name)
        result_image = cv2.resize(query_image, image_size)

        #colors
        red = [0,255,0]
        green = [0,0,255]

        for (_,path_to_image) in query_result:
            current_Im = cv2.imread(path_to_image)

            retr_image_name = get_filename_from_path(path_to_image)
            current_Im = cv2.resize(current_Im, image_size)
            if retr_image_name in correct_prediction_dictionary:
                if correct_prediction_dictionary[retr_image_name]:
                    currentIm = cv2.copyMakeBorder(current_Im, 10 , 10, 10, 10, cv2.BORDER_CONSTANT, value=green)
                else:
                    currentIm = cv2.copyMakeBorder(current_Im, 10 , 10, 10, 10, cv2.BORDER_CONSTANT, value=red)
            
            result_image = np.concatenate((result_image, current_Im), axis=1)

                    

        # for (_,path) in query_result:
        #     current_Im = cv2.imread(path, cv2.IMREAD_COLOR)

        #     if correct_prediction_dictionary[path]:
        #         cv2.copyMakeBorder(current_Im, 10 , 10, 10, 10, cv2.BORDER_CONSTANT, value=green)
        #     else:
        #         cv2.copyMakeBorder(current_Im, 10 , 10, 10, 10, cv2.BORDER_CONSTANT, value=red)
            
        #     current_Im = cv2.resize(current_Im, (200,200))
        #     result_image = np.concatenate((result_image, current_Im), axis=1)

        # display the result
        cv2.imshow("query result", result_image)
        cv2.waitKey(0)
        cv2.destroyWindows("query result")

        




if __name__ == "__main__":
    while(True):
        query = Query()
        query_result = query.run()
        print("Retrieved images: ", query_result)
        correct_prediction_dictionary = query.check_code(query_result)
        print("correct_prediction_dictionary:")
        print(correct_prediction_dictionary)
        query.visualize_result(query_result, correct_prediction_dictionary)
