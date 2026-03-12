import numpy as np
from typing import Dict, Tuple, Set, List, Optional, Callable, Any
import matplotlib.pyplot as plt
from numpy import floating
from numpy.ma.core import diag

SEED = 120
np.random.seed(SEED)

class Graph:
    def __init__(self):
        self.vertices = {}    # uid -> 3x3 matrix (Xi)
        self.edges = {}       # (u, v) tuple -> 3x3 matrix (Zij)
        self.adj = {}         # adjacency list for traversal: uid -> { neighbor_uid }

        self.averaging_map: Dict[str, Callable[[List[np.ndarray]], np.ndarray]] = {
            "euclidean": self._averaging_euclidean,
            "direction": self._averaging_direction,
            "sphere": self._averaging_sphere
        }

    # PRIVATE ################

    def _norm_matrix(self, matrix: np.ndarray) -> np.ndarray:
        det = np.linalg.det(matrix)
        if det != 0:
            return matrix / np.cbrt(det)
        return matrix

    def _averaging_euclidean(self, estimates: List[np.ndarray]) -> np.ndarray:
        return np.mean(estimates, axis=0)

    def _averaging_direction(self, estimates: List[np.ndarray]) -> np.ndarray:
        return np.mean(estimates, axis=0) # TODO


    def _averaging_sphere(self, estimates: List[np.ndarray]) -> np.ndarray:
        h_vecs = [h.flatten() / np.linalg.norm(h.flatten()) for h in estimates]

        c = np.mean(h_vecs, axis=0)
        c /= np.linalg.norm(c)

        for _ in range(10):  # Fixed-point iteration
            weights = []
            for h in h_vecs:
                dot = np.clip(np.dot(c, h), -1.0, 1.0)
                denom = np.sqrt(1 - dot ** 2)
                weights.append(1.0 / max(denom, 1e-6))

            c = np.average(h_vecs, axis=0, weights=weights)
            c /= np.linalg.norm(c)

        return c.reshape((3, 3))



    # PUBLIC ###########

    def lsh(self) -> None:
        uids = list(self.vertices.keys())
        n = len(uids)

        adj_matrix = self.build_adj_matrix()

        identity = np.identity(n * 3)

        u, s, vh = np.linalg.svd(adj_matrix - identity, full_matrices=False)
        E = u @ np.sqrt(np.identity(len(s)) * s)

        # print("U matrix:\n", u)
        # print("Singular values:\n", s)
        # print("Vh matrix:\n", vh)

        u_hat = E[:, -3:]  # Shape: (3n, 3)


        # Extract and assign the 3x3 matrices to each vertex
        for i in range(n):
            # Slice out the 3x3 block for vertex i
            raw_xi = u_hat[i * 3:(i + 1) * 3, :]
            normalized_xi = self._norm_matrix(raw_xi)

            self.vertices[uids[i]] = normalized_xi

        print("Vertex dictionary:\n", self.vertices)


    def build_adj_matrix(self) -> np.ndarray:
        uids = list(self.vertices.keys())
        n = len(uids)

        matrix = np.zeros((3 * n, 3 * n))

        for i in range(n):
            for j in range(n):
                u = uids[i]
                v = uids[j]

                edge_matrix = self.edges.get((u, v))

                if edge_matrix is not None:
                    matrix[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = edge_matrix

        return matrix





    def add_vertex(self, uid: int, initial_proj=None) -> None:
        self.vertices[uid] = initial_proj if initial_proj is not None else np.identity(3)
        if uid not in self.adj:
            self.adj[uid] = set()

    def add_edge(self, v1_id: int, v2_id: int, rel_proj: Optional[np.ndarray] = None) -> None:
        if v1_id not in self.vertices: self.add_vertex(v1_id)
        if v2_id not in self.vertices: self.add_vertex(v2_id)

        self.edges[(v1_id, v2_id)] = rel_proj if rel_proj is not None else np.identity(3)
        self.edges[(v2_id, v1_id)] = np.linalg.inv(rel_proj)

        self.adj[v1_id].add(v2_id)
        self.adj[v2_id].add(v1_id)

    def get_vertex_proj(self, uid: int) -> Optional[np.ndarray]:
        return self.vertices.get(uid)

    def get_edge_proj(self, u: int, v: int) -> Optional[np.ndarray]:
        return self.edges.get((u, v))

    def normalize(self) -> None:

        for uid in self.vertices:
            self.vertices[uid] = self._norm_matrix(self.vertices[uid])
        for pair in self.edges:
            self.edges[pair] = self._norm_matrix(self.edges[pair])

    def get_sorted_vertices(self) -> List[int]:
        return sorted(self.vertices.keys(), key=lambda uid: len(self.adj.get(uid, set())), reverse=True)

    def synchronize(self, avg_method: str = "euclidean", max_iters: int = 1000) -> None:
        sorted_vertices = self.get_sorted_vertices()
        avg_func = self.averaging_map.get(avg_method.lower(), self._averaging_euclidean)

        for _ in range(max_iters):
            new_vertices = {}
            for i in sorted_vertices:
                neighbors_ids = self.adj.get(i, set())
                if not neighbors_ids:
                    continue

                # Compute neighbor estimates: Xi|j = Zij * Xj
                estimates = [self.edges[(i, j)] @ self.vertices[j] for j in neighbors_ids]
                avg_xi = avg_func(estimates)
                new_vertices[i] = self._norm_matrix(avg_xi)

            self.vertices.update(new_vertices)




## TESTING


def generate_random():
    A = np.random.randn(3, 3)
    return A / np.cbrt(np.linalg.det(A))


def add_noise(matrix, sigma=0.01):
    noise = np.random.normal(0, sigma, (3, 3))
    return matrix + noise


def calculate_error(graph: Graph, ground_truth: Dict[int, np.ndarray]) -> floating[Any]:
    errors = []

    # In synchronization, we find a global C such that estimated_Xi * C = true_Xi
    est_X0 = graph.get_vertex_proj(0)
    true_X0 = ground_truth[0]
    C = np.linalg.inv(est_X0) @ true_X0

    for i in graph.vertices:
        est_Xi_aligned = graph.get_vertex_proj(i) @ C
        true_Xi = ground_truth[i]

        # Vectorize and normalize to unit length for angular distance
        v1 = est_Xi_aligned.flatten() / np.linalg.norm(est_Xi_aligned)
        v2 = true_Xi.flatten() / np.linalg.norm(true_Xi)

        # Angular distance: arccos(|v1 · v2|)
        cos_theta = np.clip(np.abs(np.dot(v1, v2)), 0, 1)
        errors.append(np.arccos(cos_theta))

    return np.mean(errors)


def run_experiment(node_counts: List[int], avg_method: str = "euclidean"):
    results = []

    for n in node_counts:
        g = Graph()
        ground_truth = {}

        # 1. Generate Ground Truth
        for i in range(n):
            gt_mat = generate_random()
            ground_truth[i] = gt_mat
            g.add_vertex(i, np.identity(3))

        # 2. Generate Noisy Relative Measures
        # Zij = Xi * inv(Xj) + noise
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() > 0.3: # 70% connectivity
                    rel_ij = ground_truth[i] @ np.linalg.inv(ground_truth[j])
                    noisy_rel = add_noise(rel_ij, sigma=0.05)
                    g.add_edge(i, j, noisy_rel)

        # 3. Synchronize
        g.normalize()
        g.synchronize(avg_method=avg_method, max_iters=100)

        # 4. Calculate Mean Angular Error
        error_rad = calculate_error(g, ground_truth)
        results.append(np.degrees(error_rad))
        print(f"Nodes: {n:3} | MAE: {results[-1]:.4f}°")

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(node_counts, results, marker='o', linestyle='-', color='b', label=f'SyncSL3 ({avg_method})')

    plt.title(f"Projectivity Synchronization Error vs. Graph Size ({avg_method})")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Error (degrees)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def run_experiment2(node_counts: List[int], avg_method: str = "euclidean"):
    results = []

    for n in node_counts:
        g = Graph()
        ground_truth = {}

        # 1. Generate Ground Truth
        for i in range(n):
            gt_mat = generate_random()
            ground_truth[i] = gt_mat
            g.add_vertex(i, np.identity(3))

        # 2. Generate Noisy Relative Measures
        # Zij = Xi * inv(Xj) + noise
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() > 0.3: # 70% connectivity
                    rel_ij = ground_truth[i] @ np.linalg.inv(ground_truth[j])
                    noisy_rel = add_noise(rel_ij, sigma=0.05)
                    g.add_edge(i, j, noisy_rel)

        # 3. Synchronize
        g.normalize()
        g.lsh()






def main_paper():
    # node_range = [10, 20, 30, 40, 50, 75, 100]
    node_range = [10]
    run_experiment2(node_range, avg_method="direction")


if __name__ == "__main__":
    main_paper()