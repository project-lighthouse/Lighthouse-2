import cv2

# Load the image and convert it to grayscale.
image1 = cv2.imread("../samples/products-front-back/product-1-front.jpg")
image2 = cv2.imread("../samples/products-front-back/product-1-front-1.jpg")

grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize the ORB descriptor, then detect keypoints and extract local invariant descriptors from the image.
detector = cv2.ORB_create()

(keypoints1, descriptors1) = detector.detectAndCompute(grayImage1, None)
(keypoints2, descriptors2) = detector.detectAndCompute(grayImage2, None)

print("Keypoints: {}, Descriptors: {}".format(len(keypoints1), descriptors1.shape))
print("Keypoints: {}, Descriptors: {}".format(len(keypoints2), descriptors2.shape))

# Match the features.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

matches = sorted(matches, key = lambda x:x.distance)

print("Number of matches: {}".format(len(matches)))

# cv2.drawMatchesKnn expects list of lists as matches.
image3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[1:500], None, flags=2)
cv2.imshow("ORB matching", image3)
cv2.waitKey(0)