# reduces the dimension like say from 10D to 3D or 2D
# Helps with visualization, image compression, noise reduction, and speeding up algorithms.
# Reduces features while retaining maximum variance (information).
# It has a set of mathematical operations (matrices)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Prepare DataFrames
df_original = pd.DataFrame(X_scaled, columns=feature_names)
df_original['Species'] = [target_names[i] for i in y]

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Species'] = [target_names[i] for i in y]

# Colors for plotting
colors = ['r', 'g', 'b']

# Plot before vs after PCA
plt.figure(figsize=(14,6))

# Before PCA (first two features)
plt.subplot(1,2,1)
for species, color in zip(target_names, colors):
    plt.scatter(
        df_original[df_original['Species']==species][feature_names[0]],
        df_original[df_original['Species']==species][feature_names[1]],
        label=species,
        c=color,
        s=50
    )
plt.title("Before PCA (Sepal Length vs Sepal Width)")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.grid(True)

# After PCA
plt.subplot(1,2,2)
for species, color in zip(target_names, colors):
    plt.scatter(
        df_pca[df_pca['Species']==species]['PC1'],
        df_pca[df_pca['Species']==species]['PC2'],
        label=species,
        c=color,
        s=50
    )
plt.title("After PCA (PC1 vs PC2)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)

plt.show()
