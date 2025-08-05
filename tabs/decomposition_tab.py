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
    X_ = (X - X.mean()) / X.std()
    coords = {"features": X.columns}

    with pm.Model(coords=coords) as model:
        sigma = pm.Exponential("sigma", 1.0)
        beta = pm.Normal("beta", mu=0, sigma=1, dims="features")
        intercept = pm.Normal("intercept", mu=0, sigma=1)

        mu = intercept + pm.math.dot(X_, beta)
        likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=(y - y.mean()) / y.std())

        trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=True)
    return trace, X.columns, X_, y

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
        X = df_filtered[features]
        y = df_filtered[target]

        if st.session_state.model_choice == "XGBoost/LightGBM with SHAP":
            model = xgb.XGBRegressor(objective="reg:squarederror")
            model.fit(X, y)
            explainer = shap.Explainer(model)
            shap_values = explainer(X)

            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(bbox_inches='tight')

            if st.button("Export SHAP Values CSV"):
                shap_df = pd.DataFrame(shap_values.values, columns=features)
                csv = shap_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download SHAP CSV", csv, f"shap_{target}.csv")
        else:
            with st.spinner("Running Bayesian model..."):
                trace, feature_names, X_scaled, y_vec = run_bayesian_model(X, y)

                beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
                intercept_mean = trace.posterior["intercept"].mean().values.item()
                sigma_mean = trace.posterior["sigma"].mean().values.item()

                contributions_real = X.mean().values * beta_mean
                predicted_total = contributions_real.sum() + intercept_mean
                unexplained_real = y.mean() - predicted_total

                results_df = pd.DataFrame({
                    "Feature": list(feature_names) + ["Intercept", "Unexplained"],
                    "Contribution": list(contributions_real) + [intercept_mean, unexplained_real]
                })
                results_df["% of Total"] = 100 * results_df["Contribution"] / results_df["Contribution"].sum()

                st.dataframe(results_df.style.format({"Contribution": "{:.2f}", "% of Total": "{:.1f}%"}))

                fig = px.bar(results_df, x="Feature", y="% of Total", text="% of Total",
                             title=f"Contribution Share to {target} (Bayesian Model)")
                st.plotly_chart(fig, use_container_width=True)

                if st.button("Export Posterior Summary CSV"):
                    summary_df = az.summary(trace, var_names=["beta"]).reset_index()
                    csv = summary_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Posterior CSV", csv, f"bayesian_{target}_summary.csv")

                st.subheader("Posterior Diagnostics")
                st.pyplot(az.plot_forest(trace, var_names=["beta"], combined=True))
                st.pyplot(az.plot_trace(trace))
