# ⚡ Customer Churn Prediction Web App

An end-to-end Machine Learning web application designed to evaluate and predict customer churn risk based on account demographics and billing behavior. Built using Scikit-Learn and deployed with an interactive, modern Streamlit interface.

---

## 📌 Project Overview
Customer attrition (churn) directly impacts subscription businesses. This project provides an intuitive diagnostic dashboard that helps assess whether an account holder is likely to discontinue service.

- **Primary Goal:** Real-time binary classification of customer retention risk (`Yes` / `No`).
- **Core Stack:** Python, Scikit-Learn, Streamlit, NumPy, Joblib.

---

## ⚙️ Feature Pipeline
The predictive model processes customer records structured as follows:

| Feature Name | Type | Description / Range |
| :--- | :--- | :--- |
| `Age` | Numerical | Account holder age (10 – 100 years) |
| `Gender` | Categorical | Encoded as `1` for Female, `0` for Male |
| `Tenure` | Numerical | Duration with the provider (Months) |
| `MonthlyCharges`| Numerical | Average billing amount ($) |

All input features are scaled using standard normalization via `scaler.pkl` before passing into `model.pkl`.

---

## 🚀 Key Highlights & UI Features
- **Modern Glassmorphism Design:** Dark slate theme with subtle neon borders and responsive containers.
- **Micro-Animations:** Smooth CSS keyframe entrance animations for diagnostic badges.
- **Resource Caching:** Implements Streamlit's `@st.cache_resource` for instant model loading and zero latency during repeated inferences.

---

## 🛠️ Installation & Local Setup

### 1. Clone the repository
```bash
cd customer-churn-prediction
