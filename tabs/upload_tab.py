import streamlit as st
import pandas as pd

def upload_tab():
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if {"Country", "Month", "Orders", "Revenue"}.issubset(df.columns):
                st.session_state.data = df
                st.success("File uploaded successfully.")
            else:
                st.error("Missing required columns in the dataset.")
        except Exception as e:
            st.error(f"Error reading file: {e}")