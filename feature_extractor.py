# pylint: disable=no-member
# import the necessary packages
import numpy as np
import cv2
from PIL import Image 


class FeatureExtractor: 

	#######################################################################################################################
	# Function extract(self, image)
	# Function to extract features for an image
	# 
	# Input arguments:
	#   - [image] image: input image
	# Output argument:
	#   - [list] features: list with extracted features
	#######################################################################################################################
	def extract(self, image):
		# You can change your extraction method
		
		#features = self.image_pixels(image)
		# features = self.histogram(image)
		features = self.thumbnail_features(image)
		# features = self.extract_features_spatial(image)
		# features = self.partitionbased_histograms(image)
		
		# You can even extend features with another method
		# features.extend(self.histogram(image))
		# features.extend(self.extract_features_spatial(image))
		# features.extend(self.thumbnail_features(image))
		# features.extend(self.partitionbased_histograms(image))

		return features
	
	#######################################################################################################################
	# Function histogram(self, image)
	# Function to extract histogram features for an image
	# 
	# Tipp:
	# 	- use 'cv2.calcHist'
	#	- create a list out of the histogram (hist.tolist())
	#	- return a flatten list
	#
	# Input arguments:
	#   - [image] image: input image
	# Output argument:
	#   - [list] features: list with extracted features
	#######################################################################################################################
	def histogram(self, image):
		
		his_image = cv2.calcHist(images = [image], channels = [0], mask = None, histSize = [256], ranges = [0, 256])
		
		his_list = his_image.tolist()

		his_list = [int(item) for sublist in his_list for item in sublist]

		return his_list


	#######################################################################################################################
	# Function thumbnail_features(self, image)
	# Function to create a thumbnail of an image and return the image values (features)
	#
	# Help: 
	#   - Resize image (dim(30,30))
	#	- Create a list from the image (np array)
	#	- flatten and return the list
	# 
	# Input arguments:
	#   - [image] image: input image
	# Output argument:
	#   - [list] feature_list: list with extracted features
	#######################################################################################################################
	def thumbnail_features(self, image):
		resized_image =  cv2.resize(src = image, dsize = (30,30))
		
		resized_image_list = resized_image.tolist()

		resized_image_list = [item for sublist in resized_image_list for item in sublist]

		return resized_image_list
		

	#######################################################################################################################
	# Function extract_features_spatial(self, image, factor = 10)
	# Function to create spatial features
	#
	# Help: 
	#   - Resize image (dim(200,200))
	#	- Observe (factor * factor) image parts
	#	- calculate max, min and mean for each part and add to feature list
	# 
	# Input arguments:
	#   - [image] image: input image
	#	- [int] factor: facto to split images into parts
	# Output argument:
	#   - [list] feature_list: list with extracted features
	#######################################################################################################################
	def extract_features_spatial(self, image, factor = 10):
		resized_image = cv2.resize(image, (200, 200))

		output_list = []

		#Observation of image parts:
		for i in range(0, factor):
			for j in range(0, factor):
				roi = resized_image[i:i + int(200/factor), j:int(j+200/factor)]
		
				roi = [item for sublist in roi for item in sublist]

				min = 255
				max = 0
				sum_of_roi = 0
				mean = 0
		
				for e in roi:
					if e < min:
						min = e
					if e > max:
						max = e
					sum_of_roi += e
				
				mean = sum_of_roi/ ((200/factor)*(200/factor))
				roi_list = [min, max, mean]

				output_list.append(roi_list)

		output_list = [item for sublist in output_list for item in sublist]
		return output_list

	#######################################################################################################################
	# Function partitionbased_histograms(self, image, factor = 4):
	# Function to create partition based histograms
	#
	# Help: 
	#   - Resize image (dim(100, 100))
	#	- Observe (factor * factor) image parts
	#	- calculate a histogramm for each part and add to feature list
	# 
	# Input arguments:
	#   - [image] image: input image
	#	- [int] factor: facto to split images into parts
	# Output argument:
	#   - [list] feature_list: list with extracted features
	#######################################################################################################################
	def partitionbased_histograms(self, image, factor = 4):
		resized_image = cv2.resize(image, (100, 100))

		output_list = []

		for i in range(0, factor):
			for j in range(0, factor):
				roi = resized_image[i:i + int(100/factor), j:int(j+100/factor)]

				tmp_list = cv2.calcHist([roi], [0], None, [256], [0, 256])

				tmp_list = [item for sublist in tmp_list for item in sublist]

				output_list.append(tmp_list)

		output_list = [item for sublist in output_list for item in sublist]
		return output_list

	#######################################################################################################################
	# Function image_pixels(self, image):
	# Function to return the image pixels as features
	#
	# Example of a **bad** implementation. The use of pixels as features is highly inefficient!
	# 
	# Input arguments:
	#   - [image] image: input image
	# Output argument:
	#   - [list] feature_list: list with extracted features
	#######################################################################################################################
	def image_pixels(self, image):
		# cast image to list of lists
		features =  image.tolist()
		# flatten the list of lists
		features = [item for sublist in features for item in sublist]
		# return 
		return features
