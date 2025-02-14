import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder  # To encode categorical data

data = {
    "cars": ["BMW", "AUDI", "BUGATTI", "HYUNDAI", "BENZ"],
    "price": [70000, 60000, 50000, 40000, 30000]
}

# Store data
df = pd.DataFrame(data)

# Encode categorical data
encoder = LabelEncoder()
df["cars_name"] = encoder.fit_transform(df["cars"])

# Split data for training and testing
# df["cars_name"] returns a Series (1D array).
# df[["cars_name"]] returns a DataFrame (2D array).
X = df[["cars_name"]]  
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate error
error = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {error:.2f}")

# Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["cars"], y=df["price"], label="Actual Data")  
plt.plot(df["cars"], model.predict(df[["cars_name"]]), color="red", label="Regression Line")
plt.xlabel("Car Brands")
plt.ylabel("Price")
plt.title("Linear Regression on Car Prices")
plt.legend()
plt.xticks(rotation=45)
plt.show()
