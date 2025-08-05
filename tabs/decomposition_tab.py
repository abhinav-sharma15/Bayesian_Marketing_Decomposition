import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import pymc as pm
import numpy as np
import arviz as az
import io

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
    return trace, X_, X.columns