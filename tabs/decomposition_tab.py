import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import pymc as pm
import numpy as np
import arviz as az
import io
import plotly.express as px

def run_bayesian_model(X, y):
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    coords = {"features": X.columns}

    with pm.Model(coords=coords) as model:
        X_std_matrix = pm.MutableData("X_std_matrix", X_std.values)
        sigma = pm.Exponential("sigma", 1.0)
        beta = pm.Normal("beta", mu=0, sigma=1, dims="features")
        intercept = pm.Normal("intercept", mu=0, sigma=1)

        mu = intercept + pm.math.dot(X_std_matrix, beta)  # Ensure correct matrix shape for PyMC
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_std)

        trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=True)
    return trace, X_std, y_std, X, y

def decomposition_tab():
    df = st.session_state.data.copy()
    df = df[df["Country"].isin(st.session_state.selected_countries)]
    df["Month"] = pd.to_datetime(df["Month"], format="%d/%m/%Y", errors="coerce")
    df.dropna(subset=["Month"], inplace=True)

    features = ["Paid_Search_Traffic", "Organic_Traffic", "Email_Traffic",
        "Affiliate_Traffic", "Discount_Intensity", "Personalization_Intensity"]

    time_filter = st.slider("Select Date Range", min_value=df["Month"].min().date(),
                            max_value=df["Month"].max().date(), value=(df["Month"].min().date(), df["Month"].max().date()))

    df_filtered = df[(df["Month"] >= pd.to_datetime(time_filter[0])) &
                     (df["Month"] <= pd.to_datetime(time_filter[1]))]

    for target in st.session_state.target_choice:
        st.subheader(f"Feature Decomposition for {target}")

        # Use full dataset for Bayesian model
        X_full = df[features]
        y_full = df[target]

        trace, X_std, y_std, X_raw, y_raw = run_bayesian_model(X_full, y_full)

        # Let user choose a specific month or quarter for decomposition
        df["MonthStr"] = df["Month"].dt.strftime("%Y-%m")
        period_options = sorted(df["MonthStr"].unique())
        selected_periods = st.multiselect("Select Months for Decomposition", period_options, default=[period_options[-1]])
        df_period = df[df["MonthStr"].isin(selected_periods)]
        X = df_period[features]
        y = df_period[target]

        # Reuse full-period posterior; do not retrain model for selected months

        X_std_selected = (X - X_full.mean()) / X_full.std()  # Re-standardize using training stats

        beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
        intercept_mean = trace.posterior["intercept"].mean().values.item()

        contrib_std = X_std_selected.values * beta_mean  # Ensure proper broadcasting for contribution calculation
        contrib_real = contrib_std * y.std()

        contrib_df = pd.DataFrame(contrib_real, columns=X.columns)
        mean_contrib = contrib_df.mean()
        intercept_real = intercept_mean * y.std() + y.mean()
        predicted_mean = intercept_real + mean_contrib.sum()
        unexplained = y.mean() - predicted_mean
                
        results_df = pd.DataFrame({
            "Feature": list(X.columns) + ["Intercept", "Unexplained"],
            "Contribution": list(mean_contrib) + [intercept_real, unexplained]
        })
        results_df["% of Total"] = 100 * results_df["Contribution"] / results_df["Contribution"].sum()

        st.dataframe(results_df.style.format({"Contribution": "{:.2f}", "% of Total": "{:.1f}%"}))

        fig = px.bar(results_df, x="Feature", y="% of Total", text="% of Total",
                        title=f"Contribution Share to {target} (Bayesian Model)")
        st.plotly_chart(fig, use_container_width=True)

        # Predicted vs Actual
        y_pred = intercept_real + contrib_df.sum(axis=1)
        comparison_df = pd.DataFrame({"Actual": y.values, "Predicted": y_pred})
        st.line_chart(comparison_df)

        if st.button("Export Posterior Summary CSV"):
            summary_df = az.summary(trace, var_names=["beta"]).reset_index()
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Posterior CSV", csv, f"bayesian_{target}_summary.csv")

        st.subheader("Posterior Diagnostics")
        st.pyplot(az.plot_forest(trace, var_names=["beta"], combined=True))
        st.pyplot(az.plot_trace(trace))
