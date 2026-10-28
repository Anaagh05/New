# 🧠 Customer Churn Prediction

A machine learning project to predict customer churn using classification models. Built with Python, pandas, scikit-learn, and matplotlib.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/               # Raw CSV data
│   └── processed/         # Cleaned & encoded data
│
├── notebooks/
│   └── EDA.ipynb          # Exploratory Data Analysis
│
├── src/
│   ├── preprocess.py      # Data cleaning & feature engineering
│   ├── train.py           # Model training script
│   └── evaluate.py        # Model evaluation & metrics
│
├── models/
│   └── churn_model.pkl    # Saved trained model
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
# Step 1: Preprocess the data
python src/preprocess.py

# Step 2: Train the model
python src/train.py

# Step 3: Evaluate results
python src/evaluate.py
```

---

## 📊 Results

| Model               | Accuracy | F1 Score | AUC-ROC |
|---------------------|----------|----------|---------|
| Logistic Regression | 78.4%    | 0.74     | 0.81    |
| Random Forest       | 85.1%    | 0.82     | 0.89    |
| XGBoost             | **87.3%**| **0.85** | **0.91**|

---

## 🔍 Features Used

- Contract type, tenure, monthly charges
- Internet service, tech support, online security
- Payment method, paperless billing
- Senior citizen, dependents, partner

---

## 📦 Tech Stack

- **Python 3.10**
- **pandas**, **numpy** — data wrangling
- **scikit-learn** — ML models
- **xgboost** — gradient boosting
- **matplotlib**, **seaborn** — visualization
- **joblib** — model serialization

---

## 📄 License

MIT License © 2024 Anaagh05
