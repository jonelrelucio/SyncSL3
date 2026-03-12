import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Set, List, Optional, Callable, Any
from numpy import floating

from Graph import Graph

# --- VISION PIPELINE ---
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1


def findHomographyBetweenImages(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None, None, None

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if M is not None:
            # Extract ONLY the inliers that RANSAC agreed upon
            matchesMask = mask.ravel() == 1
            inlier_src = src_pts[matchesMask]
            inlier_dst = dst_pts[matchesMask]
            return M, inlier_src, inlier_dst

    return None, None, None


def create_graph_from_real_images(dataset_paths: List[str]):
    homography_graph = Graph()
    matches_data = {}  # Dictionary to store our point pairs for evaluation

    print(f"Building graph for {len(dataset_paths)} images...")

    for i, img1_path in enumerate(dataset_paths):
        img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
        if img1 is None: continue

        for j in range(i + 1, len(dataset_paths)):
            img2_path = dataset_paths[j]
            img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
            if img2 is None: continue

            M, pts1, pts2 = findHomographyBetweenImages(img1, img2)

            if M is not None:
                # M maps from img1 to img2. So Z_ji = M (where j=img2, i=img1)
                homography_graph.add_edge(j, i, M)
                matches_data[(i, j)] = (pts1, pts2)
                print(f"Added edge between Image {i} and Image {j}")

    return homography_graph, matches_data


def calculate_reprojection_error(graph: Graph, matches_data: Dict) -> float:
    """Calculates the mean reprojection error (in pixels) across the whole graph"""
    total_error = 0.0
    total_points = 0

    for (i, j), (pts_i, pts_j) in matches_data.items():
        X_i = graph.get_vertex_proj(i)
        X_j = graph.get_vertex_proj(j)

        if X_i is None or X_j is None:
            continue

        # The synchronized relative transformation from image i to image j
        try:
            Z_ji_sync = X_j @ np.linalg.inv(X_i)
        except np.linalg.LinAlgError:
            continue

        # Warp points from image i into image j's coordinate space
        pred_pts_j = cv.perspectiveTransform(pts_i, Z_ji_sync)

        if pred_pts_j is not None:
            # Calculate Euclidean pixel distance between prediction and actual matched point
            errors = np.linalg.norm(pred_pts_j.squeeze() - pts_j.squeeze(), axis=1)
            total_error += np.sum(errors)
            total_points += len(errors)

    if total_points == 0:
        return float('inf')

    return total_error / total_points


def create_mosaic(dataset_paths: List[str], graph: Graph, reference_idx: int = 0):
    print(f"\nCreating mosaic anchored to Image {reference_idx}...")

    # Load images in color for the final output
    images = [cv.imread(path) for path in dataset_paths]

    X_ref = graph.get_vertex_proj(reference_idx)
    if X_ref is None:
        print(f"Error: Reference image {reference_idx} is not in the graph.")
        return None

    # 1. Find the bounds of the final mosaic
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    homographies = {}

    for i, img in enumerate(images):
        if img is None: continue
        X_i = graph.get_vertex_proj(i)
        if X_i is None: continue

        # Transform from image i to the reference image
        H_i_to_ref = X_ref @ np.linalg.inv(X_i)
        H_i_to_ref = H_i_to_ref / H_i_to_ref[2, 2]  # Normalize
        homographies[i] = H_i_to_ref

        # Warp the four corners of the image to find the bounding box
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped_corners = cv.perspectiveTransform(corners, H_i_to_ref)

        x_coords = warped_corners[:, 0, 0]
        y_coords = warped_corners[:, 0, 1]

        min_x = min(min_x, np.min(x_coords))
        min_y = min(min_y, np.min(y_coords))
        max_x = max(max_x, np.max(x_coords))
        max_y = max(max_y, np.max(y_coords))

    # 2. Create a translation matrix to shift everything into positive coordinates
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    # Safety check: If synchronization failed, matrices explode and request terabytes of RAM
    if canvas_w > 15000 or canvas_h > 15000:
        print(f"Canvas too large ({canvas_w}x{canvas_h}). Synchronization likely diverged.")
        return None

    # 3. Warp and blend images onto the canvas
    mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if i not in homographies: continue

        # Combine the shift (T) with the perspective warp (H)
        H_final = T @ homographies[i]
        warped_img = cv.warpPerspective(img, H_final, (canvas_w, canvas_h))

        # Simple blending: Create a mask of the warped image and place it on the mosaic
        gray = cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

        # Extract the region from the canvas, and use bitwise operations to combine
        canvas_roi = cv.bitwise_and(mosaic, mosaic, mask=cv.bitwise_not(mask))
        mosaic = cv.add(canvas_roi, warped_img)

    return mosaic

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    dataset_path = "Alcatraz_courtyard"
    dataset = []

    # ⚠️ WARNING: Doing pairwise matching on 133 images will result in 8,778 pairs!
    # For testing, highly recommend restricting this to ~5-10 images first.
    start_img = 2313
    end_img = 2445
    for i in range(start_img, start_img+5):  # Testing on just 5 images
        dataset.append("../" + dataset_path + "/" + f"San_Francisco_{i}.jpg")

    # 1. Build Graph & extract inlier matches
    g, point_matches = create_graph_from_real_images(dataset)

    # 2. Normalize matrices (crucial for projective sync)
    g.normalize()

    # Calculate error before synchronization (optional, but good for tracking improvement)
    # Note: Before synchronization, Xi = Identity, so error will naturally be high.

    # 3. Synchronize
    print("\nSynchronizing...")
    g.synchronize(avg_method="sphere", max_iters=20)

    # 4. Evaluate Reality vs. Synchronized Graph
    mean_pixel_error = calculate_reprojection_error(g, point_matches)

    print(f"\nFinal Mean Reprojection Error: {mean_pixel_error:.2f} pixels")

    # 5. Create and view the mosaic
    # We'll use Image 0 as the anchor/center of the panorama
    result_mosaic = create_mosaic(dataset, g, reference_idx=0)

    if result_mosaic is not None:
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        result_mosaic_rgb = cv.cvtColor(result_mosaic, cv.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.title("Synchronized Homography Mosaic")
        plt.imshow(result_mosaic_rgb)
        plt.axis('off')
        plt.show()

        # Optional: Save it to your disk
        # cv.imwrite("final_mosaic.jpg", result_mosaic)