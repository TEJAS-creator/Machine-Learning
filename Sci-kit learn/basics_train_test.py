import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# built in datasets - iris
# Load dataset
iris = datasets.load_iris()

# Features (X) and Labels (y)
X = iris.data
y = iris.target

print("Shape of X:", X.shape)   # Features
print("Shape of y:", y.shape)   # Labels
print("Target names:", iris.target_names[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape[0])
print("Testing size:", X_test.shape[0])
