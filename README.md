# Customer Churn Prediction for a Utility Company

Customer churn analysis and prediction project completed as part of the [BCG Data Science & Analytics Virtual Experience Program](https://www.theforage.com/simulations/bcg/data-science-ccdz).

The project combines customer behavior, contract attributes, and electricity price dynamics to identify customers at risk of churn and translate model results into retention actions.

## Business Objective

PowerCo wants to understand why customers leave and identify customers who may churn soon. The analysis focuses on four questions:

1. Is price sensitivity associated with customer churn?
2. Which customer, contract, channel, and price features are most useful for identifying churn risk?
3. How should the model threshold be selected when missing a potential churner is more costly than contacting an additional retained customer?
4. How can churn predictions support targeted retention campaigns?

## Key Results

- The dataset contains 14,606 customers, of whom 1,419 churned. The churn rate is approximately 9.7%, creating a substantial class-imbalance problem.
- A default Random Forest achieved 90.8% accuracy but only 6.5% recall for churned customers, showing why accuracy alone is misleading for this problem.
- After class balancing and threshold tuning, the Random Forest achieved 66.2% recall and 24.7% precision.
- The tuned XGBoost model achieved 71.5% recall and 14.6% precision, identifying more potential churners at the cost of more false positives.
- The final modeling decision prioritizes recall because failing to identify a customer who is about to churn may be more costly than offering an intervention to a customer who would have stayed.

## Analytical Framework

```text
Raw customer and price data
            |
            v
Exploratory data analysis
            |
            v
Customer-level price aggregation
            |
            v
Categorical and date feature engineering
            |
            v
Correlation-based feature selection
            |
            v
Random Forest and XGBoost modeling
            |
            v
Precision-recall threshold tuning
            |
            v
Retention recommendations
```

## Observations

### Customer and behavioral patterns

- Churn is a minority-class outcome, so a model can achieve high accuracy while missing most churners.
- Consumption, forecast consumption, and margin variables are strongly right-skewed and contain long tails.
- Customers with shorter tenure show a higher tendency to churn.
- Churn rates vary across sales channels and customer origins, suggesting that acquisition and onboarding experience may matter.
- Gas subscription shows limited visible separation between churned and retained customers in the exploratory analysis.

### Price-related patterns

- Monthly average electricity prices change relatively gradually at the overall-market level.
- Customer-level price dynamics may still provide useful signal through annual and six-month price variance, December-to-January price differences, and the number of price changes experienced during the year.
- The analysis supports price sensitivity as a useful modeling hypothesis, but it does not establish a causal relationship between price changes and churn.

## Feature Engineering

The feature engineering pipeline creates customer-level model inputs from the raw tables:

- Aggregates annual and recent six-month variance for six price types.
- Calculates December-to-January off-peak energy and power price differences.
- Counts how many times each price type changed during the observation period.
- Groups low-frequency sales-channel and customer-origin categories into `OTHER` before one-hot encoding.
- Converts `has_gas` into a binary feature.
- Converts contract dates into month-based features relative to `2016-01-01`.
- Removes highly correlated numeric features using a correlation threshold above 0.9.

## Model Comparison

| Model | Accuracy | Precision | Recall | Interpretation |
|---|---:|---:|---:|---|
| Random Forest baseline | 0.908 | 0.852 | 0.065 | Strong overall accuracy, but misses most churners |
| Balanced Random Forest with threshold tuning | 0.771 | 0.247 | 0.662 | Better suited to identifying at-risk customers |
| XGBoost with F2 threshold tuning | 0.565 | 0.146 | 0.715 | Highest recall, but creates more false positives |

The evaluation uses stratified train/validation/test splits. The classification threshold is selected using the validation set rather than relying only on the default 0.5 threshold. F2-based tuning gives more weight to recall than precision.

## Feature Importance and Explainability

The modeling notebook compares four views of feature importance:

- Built-in tree importance from the balanced Random Forest.
- Built-in tree importance from XGBoost.
- Test-set permutation importance for both models, measured by the decrease in average precision.
- Optional SHAP summary plots for global and model-specific explanations.

Features that rank highly across both models are treated as more stable candidates for business interpretation. The analysis also aggregates importance by business group, including contract and tenure, acquisition channel and origin, consumption and forecast, customer economics, and price dynamics.

The interpretation is deliberately conservative: feature importance shows predictive association, not causality. A high-ranking price feature can support a price-protection campaign hypothesis, but it should be validated with controlled customer experiments.

## Conclusions

1. Churn prediction should be treated as a recall-oriented classification problem rather than an accuracy-maximization problem.
2. A default model is not sufficient for this use case because it favors the majority retention class.
3. Tenure, customer characteristics, sales channel, contract economics, and customer-level price dynamics should be considered together.
4. Price features are useful inputs, but the current analysis cannot claim that price changes are the sole or causal driver of churn.
5. Threshold selection should reflect the cost of retention outreach and the value of preventing churn.

## Actionable Retention Insights

### 1. Build a prioritized churn-risk list

Use the model probability to identify high-risk customers, then rank them by both churn probability and customer value:

```text
Retention priority = churn probability × customer value
```

This helps focus expensive interventions on customers where preventing churn has the greatest expected value.

### 2. Use differentiated interventions

- Offer targeted discounts or temporary price protection to customers showing price-related risk signals.
- Provide onboarding and bill-explanation support to newer customers.
- Assign proactive service outreach to high-value, high-risk customers.
- Use low-cost automated communication for lower-value customers with elevated predicted risk.

### 3. Monitor business outcomes, not only model metrics

After deployment, track recall, precision, PR-AUC, the number of high-risk customers, intervention conversion rate, campaign cost, and estimated revenue retained. A controlled campaign test should be used to measure whether the interventions actually reduce churn.

## Data and Outputs

| File | Description |
|---|---|
| `data/client_data.csv` | Customer attributes, consumption, forecasts, contract information, and churn label |
| `data/price_data.csv` | Monthly price records for each customer |
| `data/clean_data_after_eda.csv` | Customer data merged with annual and six-month price variance features |
| `data/my_data_for_prediction.csv` | Final engineered feature table used by the modeling notebook |
| `artifacts/random_forest_balanced.joblib` | Persisted balanced Random Forest model |
| `artifacts/xgboost_best.joblib` | Persisted best XGBoost model |
| `artifacts/model_metadata.json` | Feature schema, thresholds, split settings, and model parameters |
| `artifacts/feature_importance_comparison.csv` | Cross-model feature importance comparison |

The generated CSV files are ignored by Git because the source datasets are provided externally through the BCG virtual experience. Download the source data and place it in `data/` before running the notebooks.

## How to Run

Run the notebooks in the following order from the `notebooks/` directory:

1. `eda.ipynb` — load raw data, perform exploratory analysis, and create `clean_data_after_eda.csv`.
2. `feature_engineering.ipynb` — create model features and export `my_data_for_prediction.csv`.
3. `modeling.ipynb` — train, tune, and evaluate the churn models.

Required Python packages include:

- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib
- Jupyter Notebook

SHAP is optional for explainability plots.

## Repository Structure

```text
.
├── data/
│   ├── README.md
│   ├── client_data.csv                 # downloaded source data
│   ├── price_data.csv                  # downloaded source data
│   ├── clean_data_after_eda.csv        # generated intermediate data
│   └── my_data_for_prediction.csv      # generated modeling data
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb
├── artifacts/
│   ├── random_forest_balanced.joblib
│   ├── xgboost_best.joblib
│   ├── model_metadata.json
│   └── feature_importance_comparison.csv
├── outputs/
└── README.md
```

## Limitations and Next Steps

- The analysis is observational and does not prove that price changes cause churn.
- The notebooks do not yet persist a trained model or expose a production inference pipeline.
- The current evaluation is based on a single train/validation/test split; repeated cross-validation and time-based validation would provide a stronger estimate of generalization.
- Future work should include model explainability with SHAP, campaign-level cost optimization, probability calibration, and an A/B test of retention interventions.

## Technologies

Python · Jupyter Notebook · pandas · NumPy · seaborn · Matplotlib · scikit-learn · XGBoost
