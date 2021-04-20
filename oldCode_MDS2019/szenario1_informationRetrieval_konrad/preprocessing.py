# pylint: disable=no-member
# Dataset : https://publications.rwth-aachen.de/record/667228 
import cv2
from feature_extractor import FeatureExtractor
import glob

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grand_parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, grand_parent_dir)
import get_path

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
def get_images_paths(image_directory = get_path.ImageCLEFmed2007(), file_extensions = ["*.png", "*.jpg"]):
    image_paths = []

    for ending in file_extensions:
        pathName = os.path.join(image_directory, ending)
        image_paths.extend(glob.glob(pathName))
    
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
    # initialize object and array
    feature_list = []
    featureExtractor = FeatureExtractor()

    for i,path in enumerate(image_paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        feature_list.append( featureExtractor.extract(image) )
        # keep track while processing each image feature_list
        print(str(i+1) + "/" + str(len(image_paths)))
    
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
    assert len(feature_list) == len(image_paths)
    
    f = open(output_name,"w+")

    for i in range(0,len(image_paths)):
        f.write(image_paths[i] + ",")
        f.write( ",".join(str(x) for x in feature_list[i]) + '\n' )
    f.close()
    pass



if __name__ == '__main__':
    image_paths = get_images_paths()

    features  = create_feature_list(image_paths)

    write_to_file(features, image_paths)