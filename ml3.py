import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Housing.csv")

# Display first few rows
print(df.head())

# Select features and target
X = df[["area"]]  # Independent variable
y = df["price"]   # Dependent variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Plot actual vs predicted values
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title("Linear Regression: Price vs Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()


# This value should ideally be between 0 and 1, where 1 means perfect prediction,
#  and 0 means the model performs no better than predicting the mean.
# A negative R² means the model performs worse than a simple horizontal line at
#  the mean of the target variable (i.e., it fails to capture the relationship between price and area).

# This is a very large number, meaning that the predicted prices are significantly different from
#  the actual prices.A lower MSE indicates a better model fit.




# # creating a new dataframe
# df["live"]=1+df["area"]
# # check for missing values
# print(df.isnull().sum())
# data = df.dropna()