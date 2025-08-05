import streamlit as st
import pandas as pd
import xgboost as xgb
import pymc as pm
import numpy as np
import arviz as az
import plotly.graph_objects as go
import io

def simulator_tab():
    df = st.session_state.data.copy()
    df = df[df["Country"].isin(st.session_state.selected_countries)]
    df["Month"] = pd.to_datetime(df["Month"], format="%m/%d/%Y")

    features = ["Paid_Search_Traffic", "Organic_Traffic", "Email_Traffic",
                "Affiliate_Traffic", "Discount_Intensity", "Personalization_Intensity"]

    st.subheader("What-if Analysis Simulator")
    latest_row = df.sort_values("Month").iloc[-1]

    # Recommend month with highest target
    st.markdown("### Recommended Scenario (Best Month Benchmark)")
    best_feature_values = {}
    for target in st.session_state.target_choice:
        best_month = df.sort_values(target, ascending=False).iloc[0]
        st.markdown(f"**Best {target} Month:** {best_month['Month'].strftime('%B %Y')} with {best_month[target]:,.0f} {target.lower()}")
        st.markdown("**Recommended Inputs:**")
        for feature in features:
            st.markdown(f"- {feature}: {best_month[feature]:,.0f}")
            best_feature_values[feature] = best_month[feature]

    if st.button("Use Recommended Inputs"):
        for feature in features:
            st.session_state[feature] = best_feature_values[feature]

    user_inputs = {}
    for feature in features:
        default_val = st.session_state.get(feature, float(latest_row[feature]))
        user_inputs[feature] = st.slider(f"{feature}",
                                         min_value=0.5 * latest_row[feature],
                                         max_value=1.5 * latest_row[feature],
                                         value=default_val)

    input_df = pd.DataFrame([user_inputs])
    scenario_label = st.text_input("Scenario Label", value="My Scenario")

    for target in st.session_state.target_choice:
        if st.session_state.model_choice == "XGBoost/LightGBM with SHAP":
            model = xgb.XGBRegressor(objective="reg:squarederror")
            model.fit(df[features], df[target])
            prediction = model.predict(input_df)[0]
            st.metric(f"Predicted {target} with new inputs", f"{prediction:,.2f}")
        else:
            st.info("Running Bayesian predictive model...")
            X = df[features]
            y = df[target]
            X_scaled = (X - X.mean()) / X.std()
            input_scaled = ((input_df - X.mean()) / X.std()).values

            coords = {"features": features}
            with pm.Model(coords=coords) as model:
                sigma = pm.Exponential("sigma", 1.0)
                beta = pm.Normal("beta", mu=0, sigma=1, dims="features")
                intercept = pm.Normal("intercept", mu=0, sigma=1)

                mu = intercept + pm.math.dot(X_scaled, beta)
                pm.Normal("y", mu=mu, sigma=sigma, observed=(y - y.mean()) / y.std())

                trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=False)

                posterior_pred = trace.posterior["intercept"].values + np.tensordot(
                    trace.posterior["beta"].values, input_scaled[0], axes=[[2], [0]]
                )
                posterior_pred = posterior_pred * y.std() + y.mean()

                pred_mean = posterior_pred.mean()
                pred_ci = np.percentile(posterior_pred, [5, 95])

                st.metric(f"Predicted {target} (Bayesian)", f"{pred_mean:,.2f}", f"90% CI: [{pred_ci[0]:.2f}, {pred_ci[1]:.2f}]")

                # Counterfactual Visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Original", x=[target], y=[latest_row[target]]))
                fig.add_trace(go.Bar(name="Simulated", x=[target], y=[pred_mean]))
                fig.update_layout(title=f"Counterfactual Impact on {target}", barmode='group', yaxis_title=target)
                st.plotly_chart(fig, use_container_width=True)

                # Save scenario and download
                scenario_df = input_df.copy()
                scenario_df[target + "_Prediction"] = pred_mean
                scenario_df["Scenario"] = scenario_label
                scenario_csv = scenario_df.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download Scenario CSV for {target}", scenario_csv, file_name=f"{scenario_label}_{target}_scenario.csv")
