import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(page_title="DataTalk - Data Insights Chatbot", layout="wide")

st.title("🧠 DataTalk — Your Data Insights Chatbot")
st.write("Upload a CSV file to generate summaries and visualize relationships between columns.")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
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
