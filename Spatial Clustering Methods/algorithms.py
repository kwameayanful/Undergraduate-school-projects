import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, KDTree


class KMeans():
    def __init__(self, n_clusters=3, init='k-means++', max_iter=300, random_state=None, plot_steps=False):
        """
        KMeans clustering algorithm.

        Parameters:
        n_clusters : int, default=3
            The number of clusters to form as well as the number of centroids to generate.
        init : {'k-means++', 'random'}, default='k-means++'
            Method for initialization, defaults to 'k-means++':
            'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
            'random' : choose k observations (rows) at random from data for the initial centroids.
        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm for a single run.
        random_state : int, default=None
            Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
        plot_steps : bool, default=False
            If True, plot the clustering steps.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.plot_steps = plot_steps
        self.cluster_centers_ = None
        self.cluster_labels_ = None
        self.inertia_ = None

    def random_centroids(self, data):
        """
        Initialize centroids randomly from the data points.

        Parameters:
        data : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        centroids : array, shape (n_clusters, n_features)
            The initialized centroids.
        """
        points = data.copy()
        np.random.shuffle(points)
        return points[:self.n_clusters]

    def _init_kmeans(self, data):
        """
        Initialize centroids using the k-means++ algorithm.

        Parameters:
        data : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        centroids : array, shape (n_clusters, n_features)
            The initialized centroids.
        """
        centroids = [data[0]]
        for _ in range(1, self.n_clusters):
            distance = np.min([np.linalg.norm(data - c, axis=1)**2 for c in centroids], axis=0)
            probs = distance / distance.sum()
            cum_probs = probs.cumsum()
            r_0_1 = np.random.rand()
            for i, p in enumerate(cum_probs):
                if r_0_1 < p:
                    idx = i
                    break
            centroids.append(data[idx])
        return np.array(centroids)

    def closest_centroid(self, data, centroids):
        """
        Find the closest centroid for each data point.

        Parameters:
        data : array-like, shape (n_samples, n_features)
            The input data.
        centroids : array-like, shape (n_clusters, n_features)
            The current centroids.

        Returns:
        labels : array, shape (n_samples,)
            Index of the closest centroid for each sample.
        """
        distances = np.array([np.linalg.norm(data - c, axis=1)**2 for c in centroids])
        return np.argmin(distances, axis=0)

    def get_labels(self, data, centroids):
        """
        Get the labels for each data point based on the closest centroid.

        Parameters:
        data : array-like, shape (n_samples, n_features)
            The input data.
        centroids : array-like, shape (n_clusters, n_features)
            The current centroids.

        Returns:
        labels : array, shape (n_samples,)
            Index of the closest centroid for each sample.
        """
        return self.closest_centroid(data, centroids)

    def update_centroids(self, data, labels):
        """
        Update the centroids based on the mean of the data points assigned to each centroid.

        Parameters:
        data : array-like, shape (n_samples, n_features)
            The input data.
        labels : array, shape (n_samples,)
            Index of the closest centroid for each sample.

        Returns:
        centroids : array, shape (n_clusters, n_features)
            The updated centroids.
        """
        return np.array([data[labels == k].mean(axis=0) for k in range(self.n_clusters)])

    def is_converged(self, prev_centroids, cur_centroids):
        """
        Check if the centroids have converged.

        Parameters:
        prev_centroids : array, shape (n_clusters, n_features)
            The centroids from the previous iteration.
        cur_centroids : array, shape (n_clusters, n_features)
            The current centroids.

        Returns:
        converged : bool
            True if the centroids have converged, False otherwise.
        """
        distances = np.linalg.norm(prev_centroids - cur_centroids, axis=1)
        return np.all(distances == 0)

    def fit(self, data):
        """
        Compute k-means clustering.

        Parameters:
        data : array-like, shape (n_samples, n_features)
            The input data.
        """
        if self.init == 'k-means++':
            centroids = self._init_kmeans(data)
        elif self.init == 'random':
            centroids = self.random_centroids(data)
        
        for step in range(self.max_iter):
            prev_centroids = centroids
            labels = self.get_labels(data, centroids)
            centroids = self.update_centroids(data, labels)
            if self.plot_steps:
                self.plot(data, labels, centroids, step)
            if self.is_converged(prev_centroids, centroids):
                break
        self.cluster_centers_ = centroids
        self.cluster_labels_ = labels
        self.inertia_ = sum(np.linalg.norm(self.cluster_centers_[c] - sample)**2 for c, sample in zip(self.cluster_labels_, data))

    def fit_predict(self, data):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters:
        data : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(data)
        return self.cluster_labels_

    def predict(self, data):
        """
        Predict the closest cluster each sample in data belongs to.

        Parameters:
        data : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.closest_centroid(data, self.cluster_centers_)

    def plot(self, data, labels, centroids, step):
        """
        Plot the data points and centroids at each step of the k-means algorithm.

        Parameters:
        data : array-like, shape (n_samples, n_features)
            The input data.
        labels : array, shape (n_samples,)
            Index of the closest centroid for each sample.
        centroids : array-like, shape (n_clusters, n_features)
            The current centroids.
        step : int
            The current iteration step.
        """
        plt.figure(figsize=(5, 5))
        plt.title(f'Iteration {step}')
        for k in range(self.n_clusters):
            cluster_points = data[labels == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1])
        for centroid in centroids:
            plt.scatter(*centroid, marker='x', color='k', s=100, linewidths=2)
        plt.savefig(f'img_{step:02d}.jpeg')
        plt.show()





class DBSCAN():
    def __init__(self, eps=0.5, min_samples=5, use_kdtree=False, plot_steps=False):
        """
        DBSCAN clustering algorithm.

        Parameters:
        eps : float, default=0.5
            The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples : int, default=5
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        use_kdtree : bool, default=False
            If True, use KDTree for neighborhood queries. If False, use BallTree.
        plot_steps : bool, default=False
            If True, plot the clustering steps.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.use_kdtree = use_kdtree
        self.plot_steps = plot_steps
        self.labels_ = None

    def fit(self, X):
        """
        Perform DBSCAN clustering from features or distance matrix.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data.
        """
        self.labels_ = np.full(X.shape[0], -1)  # Initialize all points as noise
        cluster_id = 0
        
        # Build KDTree or BallTree for efficient neighborhood queries
        if self.use_kdtree:
            tree = KDTree(X)
        else:
            tree = BallTree(X)
        
        for i in range(X.shape[0]):
            if self.labels_[i] != -1:  # Skip points that are already assigned
                continue
            
            neighbors = self.range_query(tree, X, i)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark as noise
            else:
                cluster_id += 1
                self.expand_cluster(tree, X, i, neighbors, cluster_id)
                if self.plot_steps:
                    self.plot(X, self.labels_, cluster_id)
        
        return self
    
    def range_query(self, tree, X, point_index):
        """
        Query the tree to find all points within eps distance from the given point.

        Parameters:
        tree : BallTree or KDTree
            The tree structure for efficient neighborhood queries.
        X : array-like, shape (n_samples, n_features)
            The input data.
        point_index : int
            The index of the point for which to find neighbors.

        Returns:
        neighbors : array, shape (n_neighbors,)
            The indices of all points within eps distance from the given point.
        """
        return tree.query_radius(X[point_index:point_index+1], r=self.eps)[0]
    
    def expand_cluster(self, tree, X, point_index, neighbors, cluster_id):
        """
        Expand the cluster from the given point.

        Parameters:
        tree : BallTree or KDTree
            The tree structure for efficient neighborhood queries.
        X : array-like, shape (n_samples, n_features)
            The input data.
        point_index : int
            The index of the point from which to expand the cluster.
        neighbors : array, shape (n_neighbors,)
            The indices of all points within eps distance from the given point.
        cluster_id : int
            The ID of the current cluster.
        """
        self.labels_[point_index] = cluster_id
        i = 0
        while i < len(neighbors):
            ni = neighbors[i]
            if self.labels_[ni] == -1:  # if the neighbor is noise
                self.labels_[ni] = cluster_id
                ni_neighbors = self.range_query(tree, X, ni)
                if len(ni_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, ni_neighbors))
            elif self.labels_[ni] == 0:  # if the neighbor is unvisited
                self.labels_[ni] = cluster_id
            i += 1
    
    def fit_predict(self, X):
        """
        Perform DBSCAN clustering from features or distance matrix and return cluster labels.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        labels : array, shape (n_samples,)
            Cluster labels for each point in the dataset given to fit().
        """
        self.fit(X)
        return self.labels_

    def plot(self, X, labels, cluster_id):
        """
        Plot the data points and current clusters.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data.
        labels : array, shape (n_samples,)
            Cluster labels for each point.
        cluster_id : int
            The current cluster ID being processed.
        """
        plt.figure(figsize=(5, 5))
        plt.title(f'Cluster {cluster_id}')
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                color = 'k'
                marker = 'x'
            else:
                color = plt.cm.Spectral(label / (len(unique_labels) - 1))
                marker = 'o'
            class_member_mask = (labels == label)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=color, marker=marker, s=50, edgecolor='k')
        plt.savefig(f'cluster_{cluster_id:02d}.jpeg')
        plt.show()