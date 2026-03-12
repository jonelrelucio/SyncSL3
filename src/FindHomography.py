import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Set, List, Optional, Callable, Any

import Graph

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

    # DRAWING MATCHES
    #img3 = cv.drawMatches(img1, kp1, img2, kp2, good[:20], None, **draw_params)
    #plt.imshow(img3, 'gray')
    #plt.show()

    return M if matchesMask is not None else None

def create_graph(dataset):
    homography_graph = Graph.Graph()
    for i, img1_path in enumerate(dataset):
        for j, img2_path in enumerate(dataset):
            if j <= i:
                continue
            img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
            img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
            M = findHomographyBetweenImages(img1, img2)
            if M is not None:
                homography_graph.add_vertex(img1_path[-8:-4])
                homography_graph.add_vertex(img2_path[-8:-4])
                homography_graph.add_edge(img1_path[-8:-4], img2_path[-8:-4], M)


dataset_path = "Alcatraz_courtyard"
dataset = []
for i in range(2313, 2446):
    dataset.append("../" + dataset_path + "/" + f"San_Francisco_{i}.jpg")

create_graph(dataset)