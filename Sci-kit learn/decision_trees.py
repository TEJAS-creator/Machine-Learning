from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# Import seaborn
import seaborn as sns

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)

print("Accuracy (Decision Tree):", accuracy_score(y_test, y_pred))

plt.figure(figsize=(12,8))
tree.plot_tree(dt, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy (Random Forest):", accuracy_score(y_test, y_pred_rf))


feature_importances = pd.Series(rf.feature_importances_, index=iris.feature_names)
print("Feature Importances:\n", feature_importances.sort_values(ascending=False))

plt.figure(figsize=(12,8))
sns.barplot(x=feature_importances.index, y=feature_importances.values)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
