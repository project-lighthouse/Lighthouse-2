import cv2

# Load the image and convert it to grayscale.
image1 = cv2.imread("../samples/products-front-back/product-1-front.jpg")
image2 = cv2.imread("../samples/products-front-back/product-1-front-2.jpg")

grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize the ORB descriptor, then detect keypoints and extract local invariant descriptors from the image.
detector = cv2.ORB_create(nfeatures=10000)

(keypoints1, descriptors1) = detector.detectAndCompute(grayImage1, None)
(keypoints2, descriptors2) = detector.detectAndCompute(grayImage2, None)

for keypoint in keypoints1:
    print("Angle {}, coordinates: {}, response: {} and size: {}".format(
        keypoint.angle, keypoint.pt, keypoint.response, keypoint.size))

print("Keypoints: {}, Descriptors: {}".format(len(keypoints1), descriptors1.shape))
print("Keypoints: {}, Descriptors: {}".format(len(keypoints2), descriptors2.shape))

# Match the features.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

#matches = sorted(matches, key=lambda x: x.distance)

#for match in matches:
#    print("Match distance {}, trainIdx: {}, queryIdx: {} and imgIdx {}".format(match.distance, match.trainIdx, match.queryIdx, match.imgIdx))

print("Number of matches: {}".format(len(matches)))
print("Number of good matches: {}".format(len(good)))

# cv2.drawMatchesKnn expects list of lists as matches.
image3 = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good, None, flags=2)
cv2.imshow("ORB matching", image3)
cv2.waitKey(0)