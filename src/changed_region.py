# pylint: disable=no-member
from __future__ import print_function
from __future__ import division
import cv2
import numpy as np

# Aggressively downsampling and blurring the two images minimizes
# differences resulting from small camera movement. It makes the
# algorithm faster and makes it work much better as well
SCALE = 8  # how much we scale down by in each dimension
BLUR_RADIUS = (25, 25)

def get_changed_region(image1, image2):
    # Shrink the two images. This, along with the blur that follows
    # will hopefully prevent us from picking up small changes that
    # come from camera motion
    img1 = cv2.resize(image1, None, fx=1/SCALE, fy=1/SCALE)
    img2 = cv2.resize(image2, None, fx=1/SCALE, fy=1/SCALE)

    # There is lots of commented out code in this algorithm because
    # it is useful to uncomment when tuning and testing.
    # cv2.imwrite('small1.png', img1);
    # small1 = img1.copy()

    img1 = cv2.GaussianBlur(img1, BLUR_RADIUS, 0)
    img2 = cv2.GaussianBlur(img2, BLUR_RADIUS, 0)
    # cv2.imwrite('blurred.png', img1);

    # We can convert to gray and then take the difference between
    # the images, but if we do that then we don't see differences where
    # the foreground and background are different colors that convert to
    # the same shade of gray.
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img1 = cv2.absdiff(img1, img2)
    # img2 = None

    # Instead of converting to gray we add the differences in the 3 channels
    b1, g1, r1 = np.split(img1, 3, 2)
    b2, g2, r2 = np.split(img2, 3, 2)
    img1 = cv2.absdiff(b1, b2) // 3 \
           + cv2.absdiff(g1, g2) // 3 \
           + cv2.absdiff(r1, r2) // 3
    img2 = None
    # cv2.imwrite('graydiff.png', img1);

    # And threshold to convert to a binary image
    _, img1 = cv2.threshold(img1, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite('thresholded.png', img1);

    # Find the contours of the thresholded image
    _, contours, _ = cv2.findContours(img1,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    # Make a list of contours that are above average area
    areas = [cv2.contourArea(c) for c in contours]
    averageArea = np.average(areas)
    large_contours = []
    for i, a in enumerate(areas):
        if a >= averageArea:
            large_contours.append(contours[i])

    # At this point, we could loop through the small contours and
    # look for ones that are close to (or overlap) this bounding box
    # of the large contours and add those in. But it could be that just
    # taking the bounding box will be good enough

    # Turn the big ones into one big messy "region of interest" contour
    roi = np.concatenate(large_contours)

    # Find the smallest rotated rectangle that holds the ROI
    minrect = cv2.minAreaRect(roi)

    # Draw various things to show the contours
    # (x, y, w, h) = cv2.boundingRect(roi)
    # minboxpts = np.int0(cv2.boxPoints(minrect))
    # cv2.drawContours(small1, [minboxpts], -1, (0, 0, 255))
    # cv2.rectangle(small1, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.drawContours(small1, [roi], -1, (0,255,0))
    # cv2.drawContours(small1, [cv2.convexHull(roi)], -1, (0,255,255))
    # cv2.imwrite('contours.png', small1)

    # Rescale the rotated rectangle back to full-size
    boxcenter = (minrect[0][0] * SCALE, minrect[0][1] * SCALE)
    boxsize = (int(minrect[1][0] * SCALE), int(minrect[1][1] * SCALE))
    boxangle = minrect[2]

    # Create a transformation matrix that will extract just the
    # rotated rectangle from the original image
    # Start with a 2x3 rotation matrix
    rotmat = cv2.getRotationMatrix2D(boxcenter, boxangle, 1)
    # Add a third row so we can multiply it
    rotation = np.float32([rotmat[0], rotmat[1], [0, 0, 1]])
    # Create a translation matrix from scratch
    translation = np.float32([[1, 0, -(boxcenter[0]-boxsize[0]/2)],
                              [0, 1, -(boxcenter[1]-boxsize[1]/2)],
                              [0, 0, 1]])
    # Combine the translation and rotation by matrix multiplication
    matrix = np.matmul(translation, rotation)
    # And convert back to a 2x3 matrix
    matrix = matrix[:2]

    # Extract just the part of the image that changed
    changed = cv2.warpAffine(image1, matrix, boxsize,
                             borderMode=cv2.BORDER_REPLICATE)

    # And return it
    return changed

# if __name__ == "__main__":
#     import time
#     from camera import Camera
#     import audioutils

#     audioutils.ALSA_SPEAKER = 'plughw:1'
#     beep = audioutils.makebeep(400, 1)

#     camera = Camera(0, 1280,960,15)
#     camera.start()
#     time.sleep(3)  # camera warm up time
#     audioutils.play(beep)
#     img1 = camera.capture()
#     audioutils.play(beep)
#     time.sleep(1)
#     img2 = camera.capture()
#     audioutils.play(beep)

#     start = time.time()
#     changed = get_changed_region(img1, img2)
#     print("Extracted changed region in:", time.time()-start)

#     cv2.imwrite('image1.png',img1)
#     cv2.imwrite('image2.png',img2)
#     cv2.imwrite('changed.png',changed)

# #    cv2.imwrite('changed.png',
# #                get_changed_region(cv2.imread("image1.png"),
# #                                   cv2.imread("image2.png")))
