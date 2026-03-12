import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Set, List, Optional, Callable, Any

from Graph import Graph

# --- VISION PIPELINE ---
# Minimum number of matched keypoints required to calculate a valid homography
MIN_MATCH_COUNT = 20
# OpenCV constant to specify the KD-Tree algorithm for FLANN matching
FLANN_INDEX_KDTREE = 1


def findHomographyBetweenImages(img1, img2):
    """
    Detects features in two images and computes the relative Homography matrix mapping img1 to img2.

    Algorithm breakdown:
    1. SIFT: Extracts scale and rotation-invariant keypoints/descriptors.
    2. FLANN: Quickly finds nearest-neighbor descriptor matches between the two images.
    3. Lowe's Ratio Test: Filters out ambiguous matches.
    4. RANSAC: Robustly estimates the homography while rejecting outliers (bad matches).

    Args:
        img1 (numpy.ndarray): The source image (grayscale).
        img2 (numpy.ndarray): The destination image (grayscale).

    Returns:
        M (numpy.ndarray or None): The 3x3 Homography matrix mapping img1 to img2.
        inlier_src (numpy.ndarray or None): The (x, y) coordinates of valid keypoints in img1.
        inlier_dst (numpy.ndarray or None): The (x, y) coordinates of valid keypoints in img2.
    """
    # Initialize SIFT (Scale-Invariant Feature Transform)
    sift = cv.SIFT_create()

    # Detect keypoints (locations) and descriptors (mathematical representations of the area around keypoints)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Safety check: ensure both images have enough features
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None, None, None

    # FLANN (Fast Library for Approximate Nearest Neighbors) is much faster than brute-force
    # matching for high-dimensional descriptors like SIFT.
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Higher checks = more accurate but slower
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Find the 2 best matches for each descriptor (k=2)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    # Lowe's Ratio Test: A good match should be significantly closer than the second-best match.
    # This filters out features that look too similar to multiple places (e.g., repeating patterns).
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # We need at least 4 points to compute a homography, but setting a higher threshold (like 10)
    # ensures better mathematical stability.
    if len(good) > MIN_MATCH_COUNT:
        # Extract the (x, y) coordinates for the good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # RANSAC (Random Sample Consensus) randomly selects subsets of points to calculate the
        # homography, finding the matrix that results in the highest number of "inliers" (points
        # that fit the model). The '5.0' is the pixel tolerance for a point to be an inlier.
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if M is not None:
            # The 'mask' array contains 1s for inliers and 0s for outliers.
            # We extract ONLY the inliers that RANSAC agreed upon to use later for error evaluation.
            matchesMask = mask.ravel() == 1
            inlier_src = src_pts[matchesMask]
            inlier_dst = dst_pts[matchesMask]
            return M, inlier_src, inlier_dst

    # Return Nones if not enough matches were found or RANSAC failed
    return None, None, None


def create_graph_from_real_images(dataset_paths: List[str]):
    """
    Iterates through all unique pairs of images in the dataset to build a
    Projectivity Synchronization graph.

    Args:
        dataset_paths (List[str]): List of file paths to the images.

    Returns:
        homography_graph (Graph): The graph containing nodes (images) and edges (relative homographies).
        matches_data (Dict): A dictionary mapping an edge tuple (i, j) to the RANSAC inlier points.
    """
    homography_graph = Graph()
    matches_data = {}  # Dictionary to store our point pairs for evaluation

    print(f"Building graph for {len(dataset_paths)} images...")

    # Pairwise matching: compares every image to every other subsequent image
    for i, img1_path in enumerate(dataset_paths):
        img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
        if img1 is None: continue

        for j in range(i + 1, len(dataset_paths)):
            img2_path = dataset_paths[j]
            img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
            if img2 is None: continue

            # Calculate the relative homography (Z_ji). Takes approx 600-700ms per pair.
            M, pts1, pts2 = findHomographyBetweenImages(img1, img2)

            if M is not None:
                # M maps from img1 to img2. So Z_ji = M (where j=img2, i=img1)
                homography_graph.add_edge(j, i, M)
                matches_data[(i, j)] = (pts1, pts2)
                print(f"Added edge between Image {i} and Image {j}")

    return homography_graph, matches_data


def calculate_reprojection_error(graph: Graph, matches_data: Dict) -> float:
    """
    Calculates the mean reprojection error (in pixels) to evaluate the accuracy of the
    synchronized graph. It does this by taking points from image I, warping them through
    the synchronized absolute matrices to image J, and measuring the pixel distance
    to the actual matched point in image J.

    Args:
        graph (Graph): The graph containing synchronized absolute homographies (X_i).
        matches_data (Dict): Dictionary containing the true matched coordinates.

    Returns:
        float: The average error in pixels. Lower is better.
    """
    total_error = 0.0
    total_points = 0

    for (i, j), (pts_i, pts_j) in matches_data.items():
        X_i = graph.get_vertex_proj(i)  # Absolute homography of image i
        X_j = graph.get_vertex_proj(j)  # Absolute homography of image j

        if X_i is None or X_j is None:
            continue

        # Reconstruct the relative transformation from image i to image j using the
        # globally synchronized absolute transformations. Math: Z_ji ≈ X_j * X_i^-1
        try:
            Z_ji_sync = X_j @ np.linalg.inv(X_i)
        except np.linalg.LinAlgError:
            continue

        # Warp the original keypoints from image i into image j's coordinate space
        # using our newly synchronized matrix.
        pred_pts_j = cv.perspectiveTransform(pts_i, Z_ji_sync)

        if pred_pts_j is not None:
            # Calculate Euclidean pixel distance between the predicted location and actual location
            errors = np.linalg.norm(pred_pts_j.squeeze() - pts_j.squeeze(), axis=1)
            total_error += np.sum(errors)
            total_points += len(errors)

    if total_points == 0:
        return float('inf')

    return total_error / total_points


def create_mosaic(dataset_paths: List[str], graph: Graph, reference_idx: int = 0):
    """
    Stitches the images together onto a single canvas using the synchronized
    absolute homographies.

    Args:
        dataset_paths (List[str]): List of file paths to the original color images.
        graph (Graph): The graph containing the synchronized absolute homographies.
        reference_idx (int): The node index to use as the base "anchor" perspective.

    Returns:
        numpy.ndarray: The final stitched mosaic image, or None if it fails.
    """
    print(f"\nCreating mosaic anchored to Image {reference_idx}...")

    # Load images in color for the final aesthetic output
    images = [cv.imread(path) for path in dataset_paths]

    X_ref = graph.get_vertex_proj(reference_idx)
    if X_ref is None:
        print(f"Error: Reference image {reference_idx} is not in the graph.")
        return None

    # 1. First Pass: Find the outer boundaries of the final mosaic so we know
    # how big to make the blank canvas.
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    homographies = {}

    for i, img in enumerate(images):
        if img is None: continue
        X_i = graph.get_vertex_proj(i)
        if X_i is None: continue

        # The transformation to warp image i into the reference image's frame.
        # Math: H = X_ref * X_i^-1
        H_i_to_ref = X_ref @ np.linalg.inv(X_i)
        H_i_to_ref = H_i_to_ref / H_i_to_ref[2, 2]  # Normalize the bottom right element to 1
        homographies[i] = H_i_to_ref

        # Warp the four corners of the current image to see where they land on the reference plane
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped_corners = cv.perspectiveTransform(corners, H_i_to_ref)

        x_coords = warped_corners[:, 0, 0]
        y_coords = warped_corners[:, 0, 1]

        # Update the global bounding box
        min_x = min(min_x, np.min(x_coords))
        min_y = min(min_y, np.min(y_coords))
        max_x = max(max_x, np.max(x_coords))
        max_y = max(max_y, np.max(y_coords))

    # 2. Canvas creation: Open CV coordinates cannot be negative. If our images
    # warped "left" or "up" from the anchor, we must translate everything right/down.
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    # Safety check: If synchronization diverged (bad data), the matrices might attempt
    # to create a canvas larger than your computer's RAM can handle.
    if canvas_w > 15000 or canvas_h > 15000:
        print(f"Canvas too large ({canvas_w}x{canvas_h}). Synchronization likely diverged.")
        return None

    # 3. Second Pass: Actually warp and blend the images onto the created canvas
    mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if i not in homographies: continue

        # Combine the positive-shift translation (T) with the perspective warp (H)
        H_final = T @ homographies[i]

        # Warp the image
        warped_img = cv.warpPerspective(img, H_final, (canvas_w, canvas_h))

        # Simple blending technique: Create a binary mask of where the warped image exists
        gray = cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

        # Clear space on the main canvas (using the inverted mask) so images don't semi-transparently overlap
        canvas_roi = cv.bitwise_and(mosaic, mosaic, mask=cv.bitwise_not(mask))
        # Add the current image to the cleared space
        mosaic = cv.add(canvas_roi, warped_img)

    return mosaic


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    dataset_path = "Alcatraz_courtyard"
    dataset = []

    # For testing, highly recommend restricting this to ~5-10 images first.
    start_img = 2313
    end_img = 2445
    for i in range(start_img, start_img + 5):  # Testing on just 20 images
        dataset.append("../" + dataset_path + "/" + f"San_Francisco_{i}.jpg")

    # 1. Build Graph & extract inlier matches
    g, point_matches = create_graph_from_real_images(dataset)

    # 2. Normalize matrices (crucial for projective sync to manage scale ambiguity)
    g.normalize()

    # 3. Synchronize absolute transformations
    print("\nSynchronizing...")
    # 'sphere' averaging utilizes the geodesic distances on a unit sphere,
    # mapping our 3x3 matrices as vectors. Max_iters set to 20 for convergence.
    g.synchronize(avg_method="sphere", max_iters=20)

    # 4. Evaluate Reality vs. Synchronized Graph
    mean_pixel_error = calculate_reprojection_error(g, point_matches)
    print(f"\nFinal Mean Reprojection Error: {mean_pixel_error:.2f} pixels")

    # 5. Create and view the final stitched mosaic
    # We use Image 0 as the anchor/center coordinate plane
    result_mosaic = create_mosaic(dataset, g, reference_idx=0)

    if result_mosaic is not None:
        # Convert BGR (OpenCV format) to RGB (Matplotlib format) for display
        result_mosaic_rgb = cv.cvtColor(result_mosaic, cv.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.title("Synchronized Homography Mosaic")
        plt.imshow(result_mosaic_rgb)
        plt.axis('off')
        plt.show()