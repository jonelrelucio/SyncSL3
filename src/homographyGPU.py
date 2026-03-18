import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from Graph import Graph

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MIN_MATCH_COUNT = 10
LOWE_RATIO = 0.7      # Lowe's ratio test threshold
RANSAC_THRESH = 5.0   # RANSAC inlier pixel tolerance


# ---------------------------------------------------------------------------
# CUDA AVAILABILITY CHECK
# Detects at startup whether a CUDA-capable GPU is available.
# If not, the pipeline falls back to CPU transparently.
# ---------------------------------------------------------------------------
CUDA_AVAILABLE = cv.cuda.getCudaEnabledDeviceCount() > 0
print(f"[Init] CUDA available: {CUDA_AVAILABLE}")


# ---------------------------------------------------------------------------
# SIFT FEATURE EXTRACTION
# We create the SIFT detector once and reuse it for every image.
# Creating it inside the loop would re-allocate memory on every call.
# ---------------------------------------------------------------------------
_sift = cv.SIFT_create()


def extract_features(img_gray: np.ndarray) -> Tuple[tuple, np.ndarray]:
    """
    Extracts SIFT keypoints and descriptors from a grayscale image.
    SIFT runs on CPU — the CUDA version exists but is in a poorly supported
    contrib module and the gains are marginal compared to the matching step.
    """
    kp, des = _sift.detectAndCompute(img_gray, None)
    return kp, des


# ---------------------------------------------------------------------------
# CUDA BFMATCHER — created once and reused across all pair comparisons.
# BFMatcher (Brute Force) on GPU is significantly faster than FLANN on CPU
# for SIFT's 128-dim float descriptors, especially as N grows.
# FLANN uses approximate search; BFMatcher is exact but GPU-parallelized,
# so it ends up faster AND more accurate.
# ---------------------------------------------------------------------------
_matcher_gpu = cv.cuda.DescriptorMatcher_createBFMatcher(cv.NORM_L2) if CUDA_AVAILABLE else None
_matcher_cpu = cv.BFMatcher(cv.NORM_L2) if not CUDA_AVAILABLE else None


def match_descriptors(des1: np.ndarray, des2: np.ndarray):
    """
    Matches two sets of SIFT descriptors using GPU BFMatcher if available,
    falling back to CPU BFMatcher otherwise.

    Returns raw knnMatch results (k=2) for Lowe's ratio test.
    Descriptors are cast to float32 — SIFT returns float32 already but we
    enforce it here defensively since GPU matchers are strict about dtype.
    """
    if CUDA_AVAILABLE:
        # Upload descriptors to GPU memory
        des1_gpu = cv.cuda_GpuMat()
        des2_gpu = cv.cuda_GpuMat()
        des1_gpu.upload(des1.astype(np.float32))
        des2_gpu.upload(des2.astype(np.float32))
        return _matcher_gpu.knnMatch(des1_gpu, des2_gpu, k=2)
    else:
        return _matcher_cpu.knnMatch(des1.astype(np.float32),
                                     des2.astype(np.float32), k=2)


def compute_homography_from_features(
    kp1, des1, kp2, des2
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Given precomputed SIFT features for two images, computes the relative
    homography matrix using GPU-accelerated matching + RANSAC.

    Steps:
      1. GPU BFMatcher finds the 2 nearest descriptor neighbors for each feature.
      2. Lowe's ratio test discards ambiguous matches.
      3. RANSAC robustly fits a homography, rejecting outlier matches.

    Returns (H, inlier_pts_img1, inlier_pts_img2) or (None, None, None).
    """
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None, None, None

    matches = match_descriptors(des1, des2)

    # Lowe's ratio test: keep only matches where the best match is clearly
    # better than the second best. Filters repeating-pattern false positives.
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) <= MIN_MATCH_COUNT:
        return None, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC: finds the homography that maximises the number of consistent
    # inliers, discarding matches that don't fit the global transformation.
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, RANSAC_THRESH)

    if M is None:
        return None, None, None

    inlier_mask = mask.ravel() == 1
    return M, src_pts[inlier_mask], dst_pts[inlier_mask]


# ---------------------------------------------------------------------------
# GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------

def create_graph_from_real_images(dataset_paths: List[str]):
    """
    Builds the homography graph from a list of image paths.

    Optimization summary vs. original:
      - SIFT detector created once outside all loops (saves repeated allocation).
      - GPU BFMatcher replaces FLANN — faster for float descriptors at scale.
      - Matcher created once and reused (avoids repeated GPU context setup).
      - All prints removed from the inner matching loop — stdout I/O inside a
        tight loop serializes execution and is a measurable bottleneck.
      - Summary printed once at the end instead.

    Returns:
        homography_graph: Graph with edges = relative homographies.
        matches_data: Dict mapping (i, j) -> (inlier_pts_i, inlier_pts_j).
    """
    homography_graph = Graph()
    matches_data = {}

    n = len(dataset_paths)
    print(f"[Step 1/2] Extracting SIFT features for {n} images...")

    # Extract features for every image once — O(N) instead of O(N²)
    precomputed = {}
    for i, path in enumerate(dataset_paths):
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is not None:
            kp, des = extract_features(img)
            precomputed[i] = (kp, des)
        else:
            print(f"  Warning: could not load {path}")
            precomputed[i] = (None, None)

    total_pairs = n * (n - 1) // 2
    print(f"[Step 2/2] Matching {total_pairs} pairs (GPU={'yes' if CUDA_AVAILABLE else 'no'})...")

    edges_found = 0
    for i in range(n):
        kp1, des1 = precomputed[i]
        if des1 is None:
            continue
        for j in range(i + 1, n):
            kp2, des2 = precomputed[j]
            if des2 is None:
                continue

            M, pts1, pts2 = compute_homography_from_features(kp1, des1, kp2, des2)

            if M is not None:
                homography_graph.add_edge(j, i, M)
                matches_data[(i, j)] = (pts1, pts2)
                edges_found += 1

    print(f"  Done — {edges_found} edges found from {total_pairs} pairs")
    return homography_graph, matches_data, precomputed


# ---------------------------------------------------------------------------
# REPROJECTION ERROR
# ---------------------------------------------------------------------------

def calculate_reprojection_error(graph: Graph, matches_data: Dict) -> float:
    """
    Measures synchronization quality by warping keypoints from image i into
    image j using the synchronized absolute homographies and measuring the
    pixel distance to the actual matched keypoint in j.

    Lower is better. Typical good values are < 5 pixels.
    """
    total_error = 0.0
    total_points = 0

    for (i, j), (pts_i, pts_j) in matches_data.items():
        X_i = graph.get_vertex_proj(i)
        X_j = graph.get_vertex_proj(j)
        if X_i is None or X_j is None:
            continue

        try:
            # Reconstruct relative transform from synchronized absolutes:
            # Z_ji ≈ X_j @ X_i^{-1}
            Z_ji_sync = X_j @ np.linalg.inv(X_i)
        except np.linalg.LinAlgError:
            continue

        pred_pts_j = cv.perspectiveTransform(pts_i, Z_ji_sync)
        if pred_pts_j is not None:
            errors = np.linalg.norm(
                pred_pts_j.squeeze() - pts_j.squeeze(), axis=1
            )
            total_error += np.sum(errors)
            total_points += len(errors)

    return total_error / total_points if total_points > 0 else float('inf')


# ---------------------------------------------------------------------------
# MOSAIC CREATION
# ---------------------------------------------------------------------------

def create_mosaic(
    dataset_paths: List[str],
    graph: Graph,
    reference_idx: int = 0
) -> Optional[np.ndarray]:
    """
    Stitches all images onto a single canvas using synchronized homographies.

    Optimization vs. original:
      - warpPerspective runs on GPU when CUDA is available. This is the single
        biggest win in this function — warping is embarrassingly parallel
        (every output pixel is independent) and maps perfectly to GPU execution.
      - Images are loaded once into a list rather than re-read per pass.
      - Canvas boundary computation and warping are kept as separate passes
        to avoid allocating oversized intermediate buffers.
    """
    print(f"\n[Mosaic] Anchoring to image {reference_idx}...")

    images = [cv.imread(p) for p in dataset_paths]

    X_ref = graph.get_vertex_proj(reference_idx)
    if X_ref is None:
        print(f"Error: reference image {reference_idx} not in graph.")
        return None

    # --- Pass 1: compute canvas bounds ---
    # Warp each image's corners to find the full extent of the output canvas.
    homographies = {}
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for i, img in enumerate(images):
        if img is None:
            continue
        X_i = graph.get_vertex_proj(i)
        if X_i is None:
            continue

        H = X_ref @ np.linalg.inv(X_i)
        H = H / H[2, 2]   # normalize so H[2,2] == 1
        homographies[i] = H

        h, w = img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
        wc = cv.perspectiveTransform(corners, H)

        min_x = min(min_x, wc[:, 0, 0].min())
        min_y = min(min_y, wc[:, 0, 1].min())
        max_x = max(max_x, wc[:, 0, 0].max())
        max_y = max(max_y, wc[:, 0, 1].max())

    # Translate so no coordinates are negative (OpenCV requires this)
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0,      1]])

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    if canvas_w > 15000 or canvas_h > 15000:
        print(f"Canvas too large ({canvas_w}x{canvas_h}) — synchronization likely diverged.")
        return None

    # --- Pass 2: warp and composite ---
    mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if i not in homographies or img is None:
            continue

        H_final = T @ homographies[i]

        if CUDA_AVAILABLE:
            # GPU warpPerspective: each output pixel is computed independently,
            # making this embarrassingly parallel — ideal for the GPU.
            img_gpu = cv.cuda_GpuMat()
            img_gpu.upload(img)
            warped_gpu = cv.cuda.warpPerspective(img_gpu, H_final, (canvas_w, canvas_h))
            warped = warped_gpu.download()
        else:
            warped = cv.warpPerspective(img, H_final, (canvas_w, canvas_h))

        # Mask-based composite: only write pixels where this image exists,
        # leaving already-written pixels from earlier images intact.
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        inv_mask = cv.bitwise_not(mask)
        mosaic = cv.add(cv.bitwise_and(mosaic, mosaic, mask=inv_mask), warped)

    return mosaic

def diagnose_graph(dataset_paths, graph, matches_data, precomputed_features):
    """
    Prints a diagnostic report to understand why the mosaic may be sparse.
    """
    n = len(dataset_paths)
    total_pairs = n * (n - 1) // 2
    edges = len(matches_data)

    print(f"\n{'='*50}")
    print(f"GRAPH DIAGNOSTICS")
    print(f"{'='*50}")
    print(f"Images:        {n}")
    print(f"Total pairs:   {total_pairs}")
    print(f"Edges found:   {edges} ({100*edges/total_pairs:.1f}% of pairs matched)")
    print(f"{'='*50}")

    # Per-image feature count
    print("\nFeatures per image:")
    for i, path in enumerate(dataset_paths):
        kp, des = precomputed_features[i]
        name = path.split('/')[-1]
        count = len(kp) if kp is not None else 0
        in_graph = graph.get_vertex_proj(i) is not None
        print(f"  [{i}] {name}: {count} keypoints — {'IN GRAPH' if in_graph else 'NOT IN GRAPH ❌'}")

    # Per-edge inlier count
    print("\nEdge inlier counts:")
    for (i, j), (pts_i, pts_j) in matches_data.items():
        print(f"  ({i},{j}): {len(pts_i)} inliers")

    # Connectivity — which images are actually reachable
    print("\nConnectivity (adjacency):")
    for i in range(n):
        neighbors = [j for (a, b) in matches_data.keys()
                     for j in ([b] if a == i else [a] if b == i else [])]
        print(f"  [{i}] connects to: {neighbors if neighbors else 'NONE ❌'}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_path = "Alcatraz_courtyard"
    start_img = 2313

    dataset = [
        f"../{dataset_path}/San_Francisco_{i}.jpg"
        for i in range(start_img, start_img + 2)
    ]

    # 1. Build graph
    g, point_matches, precomputed_features = create_graph_from_real_images(dataset)

    # Pass precomputed features — you'll need to return them from the function too
    diagnose_graph(dataset, g, point_matches, precomputed_features)

    # 2. Normalize (manages scale ambiguity in projective synchronization)
    g.normalize()

    # 3. Synchronize
    print("\n[Sync] Running sphere synchronization...")
    g.synchronize(avg_method="sphere", max_iters=20)

    # 4. Evaluate
    error = calculate_reprojection_error(g, point_matches)
    print(f"[Eval] Mean reprojection error: {error:.2f} px")

    # 5. Mosaic
    mosaic = create_mosaic(dataset, g, reference_idx=0)

    if mosaic is not None:
        mosaic_rgb = cv.cvtColor(mosaic, cv.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.title("Synchronized Homography Mosaic")
        plt.imshow(mosaic_rgb)
        plt.axis('off')
        plt.show()