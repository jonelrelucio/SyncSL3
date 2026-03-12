import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1


def findHomographyBetweenImages(img1, img2):
    # 1. SAFETY CHECK: Ensure images actually loaded
    if img1 is None:
        raise ValueError("Error: img1 is None. Check the first file path.")
    if img2 is None:
        raise ValueError("Error: img2 is None. Check the second file path.")

    # 2. UPDATE: Use SIFT_create() for modern OpenCV versions
    sift = cv.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. SAFETY CHECK: Ensure descriptors were actually found
    if des1 is None or des2 is None:
        print("Not enough features found in one or both images.")
        return

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if M is not None:  # Ensure Homography matrix was successfully found
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)

            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        else:
            print("Homography could not be calculated.")
            matchesMask = None

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask[:20],  # draw only inliers
                       flags=2)

    img3 = cv.drawMatches(img1, kp1, img2, kp2, good[:20], None, **draw_params)

    plt.imshow(img3, 'gray')
    plt.show()


# Use absolute paths or double-check your relative paths
img1 = cv.imread('../Alcatraz_courtyard/San_Francisco_2313.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('../Alcatraz_courtyard/San_Francisco_2314.jpg', cv.IMREAD_GRAYSCALE)

findHomographyBetweenImages(img1, img2)