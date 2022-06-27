import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def main():
    # generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    x, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
    x = StandardScaler().fit_transform(x)

    # DBSCAN
    db: DBSCAN = DBSCAN(eps=0.3, min_samples=19).fit(x)
    core_sample_mask = np.zeros_like(db.labels_, dtype=bool)
    core_sample_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # number of clusters in labels, ignoring noise if present
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f'estimated number of cluster: {n_clusters}')
    print(f'estimated number of noise points: {n_noise}')
    print(f'homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}')
    print(f'completeness: {metrics.completeness_score(labels_true, labels):.3f}')
    print(f'v-measure: {metrics.v_measure_score(labels_true,labels):.3f}')
    print(f'adjusted rand index: {metrics.adjusted_rand_score(labels_true,labels):.3f}')
    print(f'adjusted mutual information: {metrics.adjusted_mutual_info_score(labels_true,labels):.3f}')
    print(f'Silhouette coefficient: {metrics.silhouette_score(x, labels):.3f}')

    # plot result
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # black used for noise
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = x[class_member_mask & core_sample_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14)
        xy = x[class_member_mask & ~core_sample_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6)
    plt.title(f'Estimated number of clusters: {n_clusters}')
    plt.show()


if __name__ == '__main__':
    main()