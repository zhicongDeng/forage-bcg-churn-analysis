# BCG Data Science - Customer Churn Analysis

Customer churn prediction for a power utility company (PowerCo), completed as part of the [BCG Data Science & Analytics Virtual Experience Program](https://www.theforage.com/simulations/bcg/data-science-ccdz) on Forage.

## Background

PowerCo is a major gas and electricity utility company. The hypothesis is that **price sensitivity** is the most influential factor driving customer churn. The goal is to investigate this hypothesis through data analysis and build a predictive model.

## Project Structure

```
├── notebooks/
│   ├── eda.ipynb                  # Exploratory Data Analysis
│   ├── feature_engineering.ipynb  # Feature Engineering
│   └── modeling.ipynb             # Predictive Modeling
│
├── data/
│   └── README.md                  # Data source and description
│
└── outputs/
```

## Notebooks

### 1. Exploratory Data Analysis (`eda.ipynb`)
- Data loading and inspection of client and price datasets
- Descriptive statistics and data type analysis
- Visualisation of churn distribution across categorical variables (sales channel, origin, gas subscription)
- Distribution and violin plots for numerical features
- Monthly price trend analysis (mean, std across all price types)

### 2. Feature Engineering (`feature_engineering.ipynb`)
- One-hot encoding of categorical variables (`channel_sales`, `origin_up`)
- Date features converted to months relative to reference date
- Log transformation of skewed consumption/forecast features
- Price change count features (how many times each price type changed over the year)
- Dec-Jan off-peak price difference features
- Correlation-based feature selection (drop features with correlation > 0.9)

### 3. Predictive Modeling (`modeling.ipynb`)
- **Random Forest** baseline (300 estimators)
- **Random Forest** with `GridSearchCV` and class balancing
- Precision-Recall threshold tuning using F-beta score (beta=2, favouring recall)
- **XGBoost** with early stopping and grid search over learning rate, subsample, colsample, max depth

## Tech Stack

- Python, Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn (RandomForestClassifier, GridSearchCV)
- XGBoost
