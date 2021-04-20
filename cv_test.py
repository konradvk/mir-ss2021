import cv2
print(cv2.__version__)

img = cv2.imread("./strauss.jpg", cv2.IMREAD_COLOR)

print('Original Dimensions : ',img.shape)

 
scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ', resized.shape)
 
cv2.imshow('Resized image', resized)
cv2.waitKey(0)