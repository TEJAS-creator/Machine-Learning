from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA


# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features for better performance (especially for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
log_reg = LogisticRegression(multi_class='ovr', solver='lbfgs')
log_reg.fit(X_train, y_train)

# Train Random Forest model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)

# Evaluate models
acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"Logistic Regression Accuracy: {acc_log_reg:.2f}")
print(f"Random Forest Accuracy: {acc_rf:.2f}")

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map species names
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
df['species'] = df['species'].map(species_map)

# Pairplot for feature visualization
sns.pairplot(df, hue='species', diag_kind='kde', palette='Set1')
plt.suptitle("Feature Distribution of Iris Dataset", y=1.02)
plt.show()
