# pylint: disable=no-member
# Dataset : https://publications.rwth-aachen.de/record/667228 
import cv2
from feature_extractor import FeatureExtractor
import glob
import sys

#######################################################################################################################
# Function get_images_paths(image_directory = "./images/", file_extensions = (".png", ".jpg"))
# Function to receive every path to a file with ending "file_extension" in directory "image_directory"
#
# Help:
#   - Iterate over every file_extension
#   - Create a string pattern 
#   - Use glob to retrieve all possible file paths (https://docs.python.org/3.7/library/glob.html )
#   - Add the paths to a list  (extend)
#   -> Remember to adapt the path variable!
# 
# Input arguments:
#   - [string] image_directory = "./images/": Path to image directory
#   - [tuple]  file_extensions = (".png", ".jpg") : Tuple of strings with the possible file extensions.
# Output argument:
#   - [list] image_paths: list of image paths (strings)
#######################################################################################################################
def get_images_paths(image_directory = "/Users/falcolentzsch/Develope/ImageCLEFmed2007_test/", file_extensions = (".png", ".jpg")):
    image_paths = []
    for e in file_extensions:
        image_paths.extend(glob.glob(image_directory + "*" + e))
        
    return image_paths

#######################################################################################################################
# Function create_feature_list(image_paths)
# Function to create a features for every image in "image_paths"
#
# Help: 
#   - Iterate over all image paths
#   - Read in the image
#   - Extract features with class "feature_extractor"
#   - Add features to a list "feature_list"
# 
# Input arguments:
#   - [list] image_paths: list of image paths (strings)
# Output argument:
#   - [list] feature_list: list with extraced features
#######################################################################################################################
def create_feature_list(image_paths):
    feature_extractor = FeatureExtractor()
    feature_list = []
    for i in image_paths:
        
        feature_list.append(feature_extractor.extract(cv2.imread(i,cv2.IMREAD_GRAYSCALE)))
        
    return feature_list

#######################################################################################################################
# Function write_to_file(features, image_paths, output_name = "index.csv")
# Function to write features into a CSV file
#
# Help: 
#   - Open file ("output_name")
#   - Iterate over all features (image wise)
#   - Create a string with all features concerning one image seperated by ","
#   - Write the image paths and features in one line in the file [format: image_path,feature_1,feature_2, ..., feature_n]
#   - Close file eventually
#
#   - Information about files http://www.tutorialspoint.com/python/file_write.htm 
# 
# Input arguments:
#   - [list] feature_list: list with extraced features
#   - [list] image_paths: list of image paths (strings)
#   - [string] output_name = "index.csv": Name of the created output file
# Output argument:
#
#######################################################################################################################
def write_to_file(feature_list, image_paths, output_name = "index.csv"):
    
    output = open(output_name, "w+")
    assert len(feature_list) == len(image_paths)

    for i in range(0,len(features)):
        output.write(image_paths[i])
        for j in range(0,len(features[i])):
            output.write("," + str(features[i][j]))
        output.write('\n')


    output.close()



if __name__ == '__main__':
    image_paths = get_images_paths()

    features  = create_feature_list(image_paths)

    write_to_file(features, image_paths)