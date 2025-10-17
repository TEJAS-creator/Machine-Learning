import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(page_title="DataTalk - Data Insights Chatbot", layout="wide")

st.title("🧠 DataTalk — Your Data Insights Chatbot")
st.write("Upload a CSV file to generate summaries and visualize relationships between columns.")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

# --- Initialize Session State ---
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False
if "df" not in st.session_state:
    st.session_state["df"] = None

if uploaded_file is not None:
    # Read CSV safely
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()
    
    # --- Generate Summary ---
    if st.button("🔍 Generate Summary"):
        st.subheader("📊 Dataset Summary")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("**Column Names:**", list(df.columns))
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
        st.write("**Basic Statistics:**")
        st.write(df.describe(include='all'))

        st.session_state["data_loaded"] = True
        st.session_state["df"] = df

# --- Column Selection & Analysis ---
if st.session_state["data_loaded"] and st.session_state["df"] is not None:
    df = st.session_state["df"]

    st.subheader("📈 Analyze Columns")
    col1 = st.selectbox("Select first column", df.columns, key="col1")
    col2 = st.selectbox("Select second column", df.columns, key="col2")

    if st.button("📊 Analyze Columns"):
        if col1 == col2:
            st.warning("⚠️ Please select two different columns for analysis.")
        else:
            st.subheader(f"📊 Analysis between `{col1}` and `{col2}`")

            # Helper function to check numeric type
            def is_numeric(dtype):
                return pd.api.types.is_numeric_dtype(dtype)

            # Determine data types
            col1_type = df[col1].dtype
            col2_type = df[col2].dtype

            # Plot based on data type combination
            if is_numeric(col1_type) and is_numeric(col2_type):
                st.write("Both columns are numeric → Scatter Plot")
                fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
                st.plotly_chart(fig, use_container_width=True)

            elif is_numeric(col1_type) and not is_numeric(col2_type):
                st.write("Numeric + Categorical → Box Plot")
                fig = px.box(df, x=col2, y=col1, title=f"{col1} distribution across {col2}")
                st.plotly_chart(fig, use_container_width=True)

            elif not is_numeric(col1_type) and is_numeric(col2_type):
                st.write("Categorical + Numeric → Bar Chart")
                fig = px.bar(df, x=col1, y=col2, title=f"{col2} by {col1}")
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.write("Both columns are categorical → Crosstab Heatmap")
                cross_tab = pd.crosstab(df[col1], df[col2])
                fig = px.imshow(cross_tab, text_auto=True, title=f"{col1} vs {col2}")
                st.plotly_chart(fig, use_container_width=True)
