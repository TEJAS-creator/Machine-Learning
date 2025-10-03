# decide clusters
# select random centroids
# assign clusters
# move centroids
# check finish

# k means clustering
# decide clusters
# select random centroids
# assign clusters
# move centroids
# check finish


from sklearn.datasets import load_iris
import numpy as np
import random
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        # Step 1: initialize random centroids
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]
        print("Initial centroids:\n", self.centroids)

        # Step 2: repeat until convergence or max_iter
        for _ in range(self.max_iter):
            cluster_group = self.assign_clusters(X)
            new_centroids = self.move_centroids(X, cluster_group)

            # stop if centroids don’t change
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        for row in X:
            distances = []
            for centroid in self.centroids:
                # Euclidean distance
                distance = np.sqrt(np.dot(row - centroid, row - centroid))
                distances.append(distance)

            # Pick the index (cluster id) of the nearest centroid
            cluster_id = int(np.argmin(distances))
            cluster_group.append(cluster_id)

        return cluster_group
    
    def move_centroids(self, X, cluster_group):
        new_centroids = []
        for k in range(self.n_clusters):
            cluster_points = X[np.array(cluster_group) == k]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(self.centroids[k])  # keep old if empty
        return np.array(new_centroids)


# Load iris dataset
data = load_iris()
X = data.data[:, :2]   # just take first 2 features for plotting
y = data.target 

# Create KMeans object and call fit_predict
kmeans = Kmeans(n_clusters=2, max_iter=100)
clusters = kmeans.fit_predict(X)

print("Cluster assignments:\n", clusters[:20])  # first 20 only

# Scatter plot
colors = ['red', 'blue']
for i in range(2):
    points = X[np.array(clusters) == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=f'Cluster {i}')

# Plot centroids
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c='black', marker='X', s=100, label='Centroids')

plt.title("KMeans Clustering (2 clusters)")
plt.show()

