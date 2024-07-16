import numpy as np
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self, n_clusters=3, init='k-means++', max_iter=300, random_state=None, plot_steps=False):
        """
        
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
        points = data.copy()
        np.random.shuffle(points)
        return points[:self.n_clusters]
    
    def _init_kmeans(self, data):
        #points = data.copy()
        #np.random.shuffle(points)
        centroids = [data[0]]
        for _ in range(1, self.n_clusters):
            distance = np.min([np.linalg.norm(data-c, axis=1)**2 for c in centroids], axis=0)
            probs = distance/(distance.sum())
            cum_probs = probs.cumsum()
            r_0_1 = np.random.rand()
            for i, p in enumerate(cum_probs):
                if r_0_1 < p:
                    idx = i
                    break
            centroids.append(data[idx])
        return np.array(centroids)
    
    def closest_centroid(self, data, centroids):
        #loop over each centroid and compute the squared distance from point
        distances = np.array([np.linalg.norm(data - c, axis=1)**2 for c in centroids])
        return np.argmin(distances, axis=0)
    
    def get_labels(self, data, centroids):
        return self.closest_centroid(data, centroids)
    
    def update_centroids(self, data, labels):
        #returns the new centroids assigned from the points closest to them
        return np.array([data[labels==k].mean(axis=0) for k in range(self.n_clusters)])
    
    def is_converged(self, prev_centroids, cur_centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [np.linalg.norm(prev_centroids - cur_centroids, axis=1)]
        return np.array(distances).sum() == 0
    
    def fit(self, data):
        # initialize centroids as k random samples from X
        if self.init == 'k-means++':
            centroids = self._init_kmeans(data)
        elif self.init == 'random':
            centroids = self.random_centroids(data)
        
        for _ in range(self.max_iter):
            #distances = self.closest_centroid(data, centroids)
            prev_centroids = centroids
            labels = self.get_labels(data, centroids)
            centroids = self.update_centroids(data, labels)
            if self.is_converged(prev_centroids, centroids):
                break
        self.cluster_centers_ = centroids
        self.cluster_labels_ = labels
        self.inertia_ = sum(np.linalg.norm(self.cluster_centers_[c] - sample)**2 for c, sample in zip(self.cluster_labels_, data))
        
    def fit_predict(self, data):
        self.fit(data)
        return self.cluster_labels_
    
    def predict(self, data):
        return self.cluster_labels_
    
    def plot(self, data,clusters, centroids, steps):
        plt.figure(figsize=(5,5))
        plt.title('Iteration {}'.format(steps))
        axes = plt.gca()
        for i, index in enumerate(clusters):
            point = data[index]
            plt.scatter(point[:,0], point[:,1])
        for points in centroids:
            plt.scatter(*points, marker='x', color='k', s=20)
        plt.savefig('img_{0:02d}.jpeg'.format(steps))
        plt.show