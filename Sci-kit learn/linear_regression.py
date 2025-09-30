import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Fake dataset: area vs price
X = np.array([500, 1000, 1500, 2000, 2500]).reshape(-1, 1)
y = np.array([150, 200, 250, 300, 350])  # prices in $1000s

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Line prediction
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, y_pred, color="red", linewidth=2, label="Best fit line")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($1000s)")
plt.legend()
plt.show()

print("Coefficient (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
