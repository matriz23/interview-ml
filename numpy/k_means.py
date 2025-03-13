"""K-means clustering"""

import numpy as np
from math import inf
from matplotlib import pyplot as plt


def data_generator():
    class_1_data = np.random.randn(100, 2) + np.array([3, 4])
    class_2_data = np.random.randn(100, 2) + np.array([10, -4])
    class_3_data = np.random.randn(100, 2) + np.array([-5, 0])
    return np.concatenate([class_1_data, class_2_data, class_3_data], axis=0)


def k_means(data, K):
    N, D = data.shape

    categories = np.zeros(N)
    centroids = np.random.randn(K, D)

    for epoch in range(100):
        for i in range(N):
            nearest_centroid = None
            nearest_centroid_dist = np.inf
            for j in range(K):
                dist_ij = np.linalg.norm(data[i] - centroids[j])
                if dist_ij < nearest_centroid_dist:
                    nearest_centroid = j
                    nearest_centroid_dist = dist_ij
            categories[i] = nearest_centroid
        for j in range(K):
            centroids[j] = np.mean(data[categories == j], axis=0)
    return categories, centroids


if __name__ == "__main__":
    data = data_generator()
    K = 3

    categories, centroids = k_means(data, K)

    plt.scatter(x=data[:, 0], y=data[:, 1], c=categories)

    plt.scatter(
        centroids[:, 0], centroids[:, 1], c="red", marker="x", s=100, label="Centroids"
    )

    plt.title("K-Means Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

