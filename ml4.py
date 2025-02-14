import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Step 1.1: Load the dataset
df = pd.read_csv('house_prices.csv')

# Step 1.2: Display the first 5 rows
print(df.head())

# Step 1.3: Examine column names and data types
print(df.info())

# Step 1.4: Get summary statistics
print(df.describe())

# Step 2.1: Handle Missing Values
print(df.isnull().sum())  # Check for missing values
df = df.dropna(subset=['price'])  # Drop rows with missing target (house price)
df = df.fillna(df.median())  # Fill missing values with median

# Step 2.2: Select Relevant Features
selected_features = ['sqft_living', 'bedrooms', 'bathrooms', 'condition', 'floors']
df = df[selected_features + ['price']]

# Step 2.3: Encode Categorical Feature
df = pd.get_dummies(df, columns=['condition'], drop_first=True)

# Step 2.4: Split the Data
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3.1 & Step 3.2: Import and Create Linear Regression Model
model = LinearRegression()

# Step 3.3: Fit the Model
model.fit(X_train, y_train)

# Step 4.1: Make Predictions
y_pred = model.predict(X_test)

# Step 4.2: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 5.1: Experiment with a Different Regression Algorithm
rf = RandomForestRegressor()
param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10, 15]}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Evaluate Random Forest Model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest Mean Squared Error: {mse_rf}')
print(f'Random Forest R-squared: {r2_rf}')
