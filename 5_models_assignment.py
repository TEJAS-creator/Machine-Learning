import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("Unified Machine Learning App")

choice = st.selectbox("Choose Analysis", 
                      ["Diabetes Prediction",
                       "Sales vs Advertising Regression",
                       "Titanic Data Cleaning & Feature Engineering",
                       "Titanic Supervised Models",
                       "Titanic Unsupervised Models (KMeans + PCA)"])


if choice == "Diabetes Prediction":
    file = st.file_uploader("Upload Diabetes CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        X = data.drop("Outcome", axis=1)
        y = data["Outcome"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        models = {"Logistic Regression": lr, "Decision Tree": dt, "Random Forest": rf}
        mchoice = st.selectbox("Select Model", list(models.keys()))
        model = models[mchoice]
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        st.write("Accuracy:", acc)
        inputs = []
        for col in X.columns:
            value = st.number_input(col, min_value=0.0)
            inputs.append(value)
        if st.button("Predict"):
            ip = scaler.transform([inputs])
            result = model.predict(ip)
            st.write("Prediction:", "Positive" if result[0] == 1 else "Negative")


if choice == "Sales vs Advertising Regression":
    file = st.file_uploader("Upload Advertising CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        X = data[["Advertising"]]
        y = data["Sales"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        st.write("Mean Squared Error:", mse)
        inp = st.number_input("Advertising Spend", min_value=0.0)
        if st.button("Predict Sales"):
            st.write("Predicted Sales:", model.predict([[inp]])[0])


if choice == "Titanic Data Cleaning & Feature Engineering":
    file = st.file_uploader("Upload Titanic CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        data["Age"] = data["Age"].fillna(data["Age"].median())
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
        data = data.drop(["Cabin", "Ticket", "Name"], axis=1)
        data = pd.get_dummies(data, drop_first=True)
        st.write("Cleaned Titanic Data")
        st.dataframe(data)


if choice == "Titanic Supervised Models":
    file = st.file_uploader("Upload Cleaned Titanic CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        X = data.drop("Survived", axis=1)
        y = data["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        pred_lr = [1 if p > 0.5 else 0 for p in pred_lr]
        acc_lr = accuracy_score(y_test, pred_lr)
        pred_dt = dt.predict(X_test)
        acc_dt = accuracy_score(y_test, pred_dt)
        st.write("Linear Regression Accuracy:", acc_lr)
        st.write("Decision Tree Accuracy:", acc_dt)


if choice == "Titanic Unsupervised Models (KMeans + PCA)":
    file = st.file_uploader("Upload Cleaned Titanic CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        X = data.drop("Survived", axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        data["Cluster"] = clusters
        st.write("Clustered Data")
        st.dataframe(data)
