import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import pymc as pm
import numpy as np
import arviz as az
import plotly.express as px

# ------------------------------
# Bayesian model (train-once, reuse posterior)
# ------------------------------

def fit_bayes_full(X: pd.DataFrame, y: pd.Series):
    """Fit a Bayesian linear model on the FULL period.
    Returns posterior and training stats needed for re-use on any subset.
    Mirrors the common Colab pattern: standardize X, y; fit; reuse betas.
    """
    # Training standardization stats
    x_mean = X.mean()
    x_std = X.std().replace(0, 1.0)  # guard
    y_mean = float(y.mean())
    y_std = float(y.std() if y.std() != 0 else 1.0)

    X_std = (X - x_mean) / x_std
    y_std_vec = (y - y_mean) / y_std

    coords = {"features": X.columns}

    with pm.Model(coords=coords) as model:
        X_std_data = pm.MutableData("X_std_data", X_std.values)
        sigma = pm.Exponential("sigma", 1.0)
        beta = pm.Normal("beta", mu=0, sigma=1, dims="features")
        intercept = pm.Normal("intercept", mu=0, sigma=1)

        mu = intercept + pm.math.dot(X_std_data, beta)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_std_vec.values)

        trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=True)

    train_stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_names": list(X.columns),
    }
    return trace, train_stats


def period_contributions_from_posterior(trace, train_stats, X_period: pd.DataFrame, y_period: pd.Series):
    """Compute per-feature contributions for a selected period using the full-period posterior.
    Steps:
      1) Standardize X_period using TRAIN stats
      2) Row-wise contrib in std space: X_std * beta_mean
      3) Convert contrib back to original units by * y_std
      4) Aggregate (mean over rows) to get average contribution for the period
      5) Optionally compute intercept & unexplained
    """
    x_mean = train_stats["x_mean"]
    x_std = train_stats["x_std"]
    y_mean = train_stats["y_mean"]
    y_std = train_stats["y_std"]
    feature_names = train_stats["feature_names"]

    # Standardize period using TRAIN stats
    X_std = (X_period - x_mean) / x_std
    X_std = X_std[feature_names]  # align columns

    # Posterior means
    beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"]).values  # (n_features,)
    intercept_mean = float(trace.posterior["intercept"].mean().values)

    # Row-wise contributions in std space
    contrib_std = X_std.values * beta_mean  # (n_rows, n_features)
    contrib_real_rows = contrib_std * y_std  # back to original y units

    # Average contribution across selected rows (period)
    mean_contrib = contrib_real_rows.mean(axis=0)  # (n_features,)

    # Intercept & unexplained (optional)
    intercept_real = intercept_mean * y_std + y_mean
    y_pred_rows = intercept_real + contrib_real_rows.sum(axis=1)
    y_period_mean = float(y_period.mean()) if len(y_period) else 0.0
    unexplained = y_period_mean - float(y_pred_rows.mean())

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": mean_contrib
    })
    meta = {
        "intercept_real": intercept_real,
        "unexplained": unexplained,
        "y_pred_mean": float(y_pred_rows.mean()) if len(y_period) else intercept_real + mean_contrib.sum(),
        "y_actual_mean": y_period_mean,
    }
    return contrib_df, meta


# ------------------------------
# Streamlit tab
# ------------------------------

def decomposition_tab():
    df = st.session_state.data.copy()
    df = df[df["Country"].isin(st.session_state.selected_countries)]
    # Robust date parsing; user confirmed dd/mm/yyyy
    df["Month"] = pd.to_datetime(df["Month"], format="%d/%m/%Y", errors="coerce")
    df.dropna(subset=["Month"], inplace=True)

    features = [
        "Paid_Search_Traffic", "Organic_Traffic", "Email_Traffic",
        "Affiliate_Traffic", "Discount_Intensity", "Personalization_Intensity"
    ]

    # Select time window for training (defaults to full span)
    st.markdown("### 1) Select training window (model fits once on full range)")
    train_range = st.slider(
        "Training period",
        min_value=df["Month"].min().date(),
        max_value=df["Month"].max().date(),
        value=(df["Month"].min().date(), df["Month"].max().date())
    )
    df_train = df[(df["Month"] >= pd.to_datetime(train_range[0])) & (df["Month"] <= pd.to_datetime(train_range[1]))]

    # Targets to loop over
    for target in st.session_state.target_choice:
        st.subheader(f"Feature Decomposition — {target}")

        # ---------------- Train once on full period ----------------
        X_full = df_train[features]
        y_full = df_train[target]

        if st.session_state.model_choice == "XGBoost/LightGBM with SHAP":
            model = xgb.XGBRegressor(objective="reg:squarederror")
            model.fit(X_full, y_full)
            explainer = shap.Explainer(model)
            shap_values = explainer(X_full)

            shap.summary_plot(shap_values, X_full, show=False)
            st.pyplot(bbox_inches="tight")

            if st.button(f"Export SHAP Values CSV — {target}"):
                shap_df = pd.DataFrame(shap_values.values, columns=features)
                csv = shap_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download SHAP CSV", csv, f"shap_{target}.csv")
            continue  # skip Bayesian UI for this target

        # Bayesian path
        with st.spinner("Fitting Bayesian model on full training window..."):
            trace, train_stats = fit_bayes_full(X_full, y_full)

        # ---------------- Period selection (months / quarters) ----------------
        st.markdown("### 2) Select period(s) to *analyze* using the trained posterior")
        df["MonthStr"] = df["Month"].dt.strftime("%Y-%m")
        month_choices = sorted(df["MonthStr"].unique())
        selected_months = st.multiselect(
            "Months to analyze",
            options=month_choices,
            default=[month_choices[-1]] if month_choices else []
        )

        # Optional: quarter selection
        df["Quarter"] = df["Month"].dt.to_period("Q").astype(str)
        quarter_choices = sorted(df["Quarter"].unique())
        use_quarter = st.checkbox("Analyze by quarter instead of months", value=False)
        selected_quarters = []
        if use_quarter:
            selected_quarters = st.multiselect(
                "Quarters to analyze",
                options=quarter_choices,
                default=[quarter_choices[-1]] if quarter_choices else []
            )

        # Build the period dataframe
        if use_quarter:
            df_period = df[df["Quarter"].isin(selected_quarters)].copy()
        else:
            df_period = df[df["MonthStr"].isin(selected_months)].copy()

        if df_period.empty:
            st.warning("No rows in the selected analysis period.")
            continue

        X_period = df_period[features]
        y_period = df_period[target]

        # ---------------- Compute contributions using posterior ----------------
        contrib_df, meta = period_contributions_from_posterior(trace, train_stats, X_period, y_period)

        # Toggle to include intercept & unexplained rows
        show_baseline = st.checkbox("Show intercept & unexplained rows", value=False, key=f"baseline_{target}")
        if show_baseline:
            baseline_rows = pd.DataFrame({
                "Feature": ["Intercept", "Unexplained"],
                "Contribution": [meta["intercept_real"], meta["unexplained"]],
            })
            results_df = pd.concat([contrib_df, baseline_rows], ignore_index=True)
        else:
            results_df = contrib_df.copy()

        # % of total (use absolute sum to avoid sign-cancel for shares if desired)
        denom = results_df["Contribution"].sum()
        if denom == 0:
            denom = 1e-9
        results_df["% of Total"] = 100 * results_df["Contribution"] / denom

        # ---------------- Render ----------------
        st.markdown("### 3) Contributions in selected period")
        st.dataframe(results_df.style.format({"Contribution": "{:.2f}", "% of Total": "{:.1f}%"}))

        fig = px.bar(results_df, x="Feature", y="% of Total", text="% of Total",
                     title=f"Contribution Share to {target} — Selected Period")
        st.plotly_chart(fig, use_container_width=True)

        # Predicted vs Actual overview
        st.markdown("#### Predicted vs Actual (mean of selected rows)")
        pvact = pd.DataFrame({
            "Metric": ["Actual Mean", "Predicted Mean"],
            "Value": [meta["y_actual_mean"], meta["y_pred_mean"]],
        })
        st.table(pvact.style.format({"Value": "{:.2f}"}))

        # Diagnostics
        with st.expander("Bayesian diagnostics (optional)"):
            st.pyplot(az.plot_forest(trace, var_names=["beta"], combined=True))
            st.pyplot(az.plot_trace(trace))
