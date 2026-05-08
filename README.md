# Insurance Loss Analytics: Predicting Loss Cost & Claim Likelihood

**USC Marshall School of Business | DSO 530 | Spring 2026**

---

## Project Overview

Insurance companies face an adverse selection problem: mispricing policies causes low-risk customers to leave for cheaper competitors, leaving the insurer with an unprofitable, high-risk pool. This project builds and validates predictive models for three key actuarial quantities using a portfolio of **39,928 auto insurance policies**.

| Target | Definition |
|--------|-----------|
| **LC** | Loss Cost per exposure unit |
| **HALC** | Historically Adjusted Loss Cost |
| **CS** | Claim Status (did the policyholder file a claim?) |

---

## My Contribution

- **SHAP Analysis**: Applied SHAP (SHapley Additive exPlanations) across all three models to identify the key drivers of insurance risk
- **Business Insights**: Translated model outputs into actionable recommendations for underwriters and pricing teams
- **Prediction Pipeline**: Built the final CS classification model (XGBoost, AUC = 0.857) and generated the merged prediction CSV

---

## Key Findings

**Behavioral variables dominate risk — outranking demographics.**

SHAP analysis revealed a consistent pattern across all three models:

| Rank | LC Driver | HALC Driver | CS Driver |
|------|-----------|-------------|-----------|
| 1 | cancel_ratio | policy_start_age_yrs | cancel_ratio |
| 2 | days_since_renewal | cancel_ratio | days_to_renewal |
| 3 | policy_start_age_yrs | days_since_renewal | policy_start_age_yrs |
| 4 | days_to_renewal | days_to_renewal | days_since_renewal |
| 5 | Net Premium (X.14) | Net Premium (X.14) | Pay Frequency (X.13) |

**Business Implication**: Insurers should price based on *how customers behave* — not just who they are. Cancellation history and renewal timing are stronger risk signals than age or vehicle type. Policies approaching renewal show elevated claim activity, suggesting strategic claiming behavior that premium-only pricing fails to capture.

---

## Model Performance

| Target | Final Model | Validation Score |
|--------|------------|-----------------|
| LC | LightGBM (Tweedie) | MSE = 254,710 |
| HALC | LightGBM (Tweedie) | MSE = 996,779 |
| CS | XGBoost Classifier | **AUC = 0.857** |

4 candidate models were benchmarked per target: Tweedie GLM, XGBoost, LightGBM, and an XGBoost+LightGBM ensemble.

---

## Technical Approach

**Data Challenges**
- 88.85% zero-claim rate → Tweedie distribution for regression targets
- 11.2% claim rate → severe class imbalance for CS classifier
- Target leakage → X.15–X.18 excluded from all feature matrices

**Feature Engineering** (28 raw variables → 33 features)
- Date columns → actionable time features: `policy_start_age_yrs`, `days_since_renewal`, `days_to_renewal`, `license_age_yrs`, `insured_age_yrs`
- Behavioral features: `cancel_ratio`, `power_to_weight`, `premium_per_policy`, `vehicle_age`

**Modeling**
- Regression: Tweedie deviance loss (handles zero-inflation + heavy tail)
- Classification: `scale_pos_weight` to handle class imbalance
- Validation: 80/20 hold-out split; models retrained on full data before test prediction
- Post-processing: LC/HALC set to 0 where CS probability < 0.5 (structural alignment)

---

## Tech Stack

```
Python | XGBoost | LightGBM | SHAP | scikit-learn | pandas | matplotlib
```

---

## Repository Structure

```
insurance-loss-analytics/
├── README.md
├── notebooks/
│   ├── Data_cleaning_and_EDA.ipynb       # Jessica & Ryan
│   ├── Regression_Models.ipynb            # Bobby
│   ├── Classification_Model.ipynb         # Anna
│   └── SHAP_Analysis_and_Prediction.ipynb # Kiki
├── outputs/
│   ├── shap_CS_summary.png
│   ├── shap_LC_summary.png
│   ├── shap_HALC_summary.png
│   └── shap_LC_HALC_bar.png
└── report/
    └── Group_5_report.pdf
```

---

## Business Recommendations

1. **Incorporate behavioral signals** — cancellation ratio and renewal timing should be standard inputs in underwriting decisions
2. **Flag near-renewal policies** for additional review; elevated claim activity near renewal boundaries suggests strategic behavior
3. **Move beyond premium-only pricing** — residual signal in behavioral variables indicates current pricing models leave risk on the table
4. **Do not transfer this model to life insurance** — Tweedie assumptions and feature engineering are auto-specific

---

*Note: Raw data not included in this repository per course data sharing policy.*
