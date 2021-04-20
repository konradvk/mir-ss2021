# pylint: disable=no-member
# import the necessary packages
import numpy as np
import cv2
 
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
		# TODO: You can change your extraction method
		features = []
		# features.extend(self.image_pixels(image))
		features.extend(self.histogram(image))
		# TODO: You can even extend features with another method
		features.extend(self.thumbnail_features(image))
		features.extend(self.extract_features_spatial(image,factor=10))
		# features.extend(self.partitionbased_histograms(image,factor=4))

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
		hist = cv2.calcHist(images = [image], channels = [0], mask = None, histSize = [256], ranges = [0,256])
		hist.tolist()
		# return hist.flatten()
		return [int(item) for sublist in hist for item in sublist]


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
		resizedIm = cv2.resize(image, (30,30))
		return ( np.asarray(resizedIm) ).flatten()
		# return [item for sublist in resizedIm for item in sublist]

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
		resizedIm = cv2.resize(src=image,dsize=(200,200))
		gridSize = int(200/factor)
		results = []
		for j in range(0,factor):
			for i in range(0,factor):
					# left,top,right,bottom
					# croppedIm = resizedIm2.crop((i*gridSize,j*gridSize,200-(i+1)*gridSize,200-(j+1)*gridSize))
				croppedIm = resizedIm[j*gridSize:(j+1)*gridSize, i*gridSize:(i+1)*gridSize]
				croppedFl = [item for sublist in croppedIm for item in sublist]
				results.extend([min(croppedFl), max(croppedFl), int( sum(croppedFl)/len(croppedFl) )])
		return results		

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
		gridSize = int(100/factor)
		resizedIm = cv2.resize(src=image,dsize=(100,100))
		results = []
		for j in range(0,factor):
			for i in range(0,factor):
				# possible edge case: if j==factor-1
				# --> make the last grid size until the array end size
				croppedIm = resizedIm[j*gridSize:(j+1)*gridSize, i*gridSize:(i+1)*gridSize]
				hist = cv2.calcHist(images = [croppedIm], channels = [0], mask = None, histSize = [256], ranges = [0,256])
				results.extend([int(item) for sublist in hist for item in sublist])
		return results

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


if __name__ == '__main__':
	# Read the test image
	# TODO change image path to a valid one
	example_image = cv2.imread("C:/Users/kvkue/Pictures/MDStestdata/ImageCLEFmed2007_test/3145.png", cv2.IMREAD_GRAYSCALE)
	# example_image = cv2.imread("C:/Users/kvkue/Pictures/MDStestdata/white.png", cv2.IMREAD_GRAYSCALE)
	# example_image = cv2.imread("C:/Users/kvkue/Dropbox/B.Sc Med Inf/Medical Data Science/feature_extractor/dog.jpg", cv2.IMREAD_GRAYSCALE)

	# Assert image read was successful
	assert example_image is not None

	# create extractor
	feature_extractor = FeatureExtractor()

	# describe the image
	features = feature_extractor.extract(example_image)

	# print the features
	print("Features: ", features)
	print("Length: ", len(features))
