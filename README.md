# Bayesian Marketing Decomposition App

This Streamlit app analyzes the impact of various marketing channels on orders and revenue over time using two modeling approaches:

- **XGBoost/LightGBM with SHAP decomposition** (for speed)
- **Bayesian Marketing Mix Modeling (MMM)** with hierarchical priors and uncertainty estimates

## Features

- Upload monthly marketing dataset (country, month, traffic sources, conversions)
- Select countries and targets (Orders / Revenue)
- Visualize trends and decompose feature impacts
- Run counterfactual simulations with a what-if analyzer
- Automatically recommend best historical scenario and simulate outcomes
- Export results as CSV

## Deployment

1. Clone the repo:
```bash
git clone https://github.com/yourusername/bayesian-mkt-app.git
cd bayesian-mkt-app
```

2. Create environment and install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Data Format Example

```csv
Country,Month,Paid_Search_Traffic,Organic_Traffic,Email_Traffic,Affiliate_Traffic,Discount_Intensity,Personalization_Intensity,Orders,Revenue
Germany,01/01/2023,12745,7987,2089,3050,0.04,0.68,4897,215460.56
```

---

© 2025 Your Company Name