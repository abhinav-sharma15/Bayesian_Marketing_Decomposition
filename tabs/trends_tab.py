import pandas as pd
import plotly.express as px
import streamlit as st

def trends_tab():
    df = st.session_state.data.copy()
    df = df[df["Country"].isin(st.session_state.selected_countries)]
    df["Month"] = pd.to_datetime(df["Month"], format="%d/%m/%Y", errors="coerce")
    df.sort_values("Month", inplace=True)

    metrics = ["Paid_Search_Traffic", "Organic_Traffic", "Email_Traffic",
               "Affiliate_Traffic", "Discount_Intensity", "Personalization_Intensity",
               "Orders", "Revenue"]

    st.subheader("Monthly Trends")
    selected_metrics = st.multiselect("Select metrics to visualize", metrics, default=["Orders", "Revenue"])

    for metric in selected_metrics:
        fig = px.line(df, x="Month", y=metric, color="Country", markers=True, title=f"{metric} Over Time")
        st.plotly_chart(fig, use_container_width=True)
