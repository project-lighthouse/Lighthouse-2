import cv2

# Load the image and convert it to grayscale.
image1 = cv2.imread("../samples/products-front-back/product-1-front.jpg")
image2 = cv2.imread("../samples/products-front-back/product-1-front-1.jpg")

#im1 = cv2.medianBlur(im1, 25)
#im2 = cv2.medianBlur(im2, 25)

grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize the AKAZE descriptor, then detect keypoints and extract local invariant descriptors from the image.
detector = cv2.AKAZE_create()

(keypoints1, descriptors1) = detector.detectAndCompute(grayImage1, None)
(keypoints2, descriptors2) = detector.detectAndCompute(grayImage2, None)

print("Keypoints: {}, Descriptors: {}".format(len(keypoints1), descriptors1.shape))
print("Keypoints: {}, Descriptors: {}".format(len(keypoints2), descriptors2.shape))

# Match the features.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)
matches = matcher.match(descriptors1, descriptors1)

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(descs1, descs2, k=2)

print("Number of matches: {}".format(len(matches)))

goodMatches = sorted(matches, key = lambda x:x.distance)

# cv2.drawMatchesKnn expects list of lists as matches.
image3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, goodMatches[1:500], None, flags=2)
cv2.imshow("AKAZE matching", image3)
cv2.waitKey(0)