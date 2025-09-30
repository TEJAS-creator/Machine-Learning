from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Classes:", model.classes_)
print("Coefficients shape:", model.coef_.shape)

y_pred = model.predict(X_test)

print("First 5 Predictions:", y_pred[:5])
print("First 5 Actual:", y_test[:5])

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

probs = model.predict_proba(X_test[:5])
print("Probabilities for first 5 samples:\n", probs)
