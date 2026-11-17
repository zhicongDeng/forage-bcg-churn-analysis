# Customer Churn Prediction for a Utility Company

Customer churn analysis completed for the [BCG Data Science & Analytics Virtual Experience Program](https://www.theforage.com/simulations/bcg/data-science-ccdz). The project combines customer behavior, contract attributes, and electricity price dynamics to identify churn risk and translate model output into retention actions.

## Key takeaways

- Churn is a minority-class problem: 1,419 of 14,606 customers churned (9.7%), so accuracy alone is not an appropriate success metric.
- After class balancing and threshold tuning, the Random Forest reached 66.2% recall and 24.7% precision; XGBoost reached 71.5% recall and 14.6% precision.
- `margin_gross_pow_ele` is the strongest shared signal. It represents the gross margin earned on a customer's power subscription and ranks first by permutation importance for both models.
- XGBoost uses a broad, composite feature signal. Random Forest has a sharper hierarchy, making it easier to explain and translate into a focused retention campaign.
- The recommended campaign model is Random Forest, with XGBoost retained as a higher-recall challenger.

## Business question

Which customers are most likely to churn, which features explain that risk, and how can the business target retention actions when missing a potential churner is more costly than contacting an additional retained customer?

## Analytical approach

```text
Raw customer and price data
        -> exploratory analysis
        -> customer-level price aggregation
        -> date and categorical feature engineering
        -> correlation-based feature selection
        -> Random Forest and XGBoost
        -> validation-based threshold tuning
        -> feature importance and SHAP analysis
        -> retention recommendations
```

The workflow uses stratified train/validation/test splits. Thresholds are selected on the validation set, with F2-based tuning to give more weight to recall. The final test set is reserved for evaluation.

## Main observations

- Consumption, forecast consumption, and margin variables are right-skewed and contain long tails.
- Shorter-tenure customers show higher churn tendency.
- Churn varies across sales channels and customer origins, indicating potential onboarding or acquisition effects.
- Customer-level price dynamics are more informative than the overall market price trend through price variance, December-to-January differences, and price-change counts.
- The analysis supports price sensitivity as a useful predictive hypothesis; it does not establish that price changes cause churn.

## Feature engineering

- Aggregate annual and recent six-month price variance for six price types.
- Calculate December-to-January off-peak energy and power price differences.
- Count price changes during the observation period.
- Group low-frequency channel and origin categories into `OTHER` before one-hot encoding.
- Convert `has_gas` to a binary feature and contract dates to month-based features relative to `2016-01-01`.
- Remove highly correlated numeric features using a correlation threshold above 0.9.

## Model comparison

| Model | Accuracy | Precision | Recall | Role |
|---|---:|---:|---:|---|
| Random Forest baseline | 0.908 | 0.852 | 0.065 | Reference only; misses most churners |
| Balanced Random Forest + threshold tuning | 0.771 | 0.247 | 0.662 | Preferred campaign model |
| XGBoost + F2 threshold tuning | 0.565 | 0.146 | 0.715 | Higher-recall challenger |

## Explainability and research conclusion

The modeling notebook compares built-in tree importance, test-set permutation importance, and positive-class SHAP values for both models.

### Shared signal

`margin_gross_pow_ele` ranks first by permutation importance for both Random Forest (approximately 0.152) and XGBoost (approximately 0.165). The two methods therefore agree that power-subscription margin and related price economics deserve the main commercial focus.

### Different model structures

- XGBoost shows an approximately exponential decline after ranking: many features contribute to a composite churn signal spanning margin, consumption, contract timing, subscribed power, and price dynamics.
- Random Forest is more concentrated: `margin_gross_pow_ele` is far ahead of the next feature, `cons_12m`, while roughly half of the features have limited or negative permutation contribution.
- The Random Forest hierarchy is easier to communicate and supports a narrower intervention design. XGBoost remains useful when higher recall and broader feature interactions are valuable.

The SHAP ranking supports the same conclusion. Random Forest is led by `margin_gross_pow_ele`, `cons_12m`, `month_activ`, `forecast_meter_rent_12m`, and `net_margin`. XGBoost is led by `margin_gross_pow_ele`, `pow_max`, `net_margin`, `month_to_end`, and `var_year_price_off_peak_var`. SHAP magnitudes should be compared within each model because the models use different output scales.

## Actionable recommendations

1. Run an A/B test for customers with high predicted churn and high `margin_gross_pow_ele`, comparing targeted price protection or tariff review with standard communication.
2. Match offers to consumption and subscribed power; use `cons_12m` and `pow_max` to distinguish customer needs.
3. Contact customers proactively around `month_to_end`, `month_activ`, and `month_renew`, especially for newer or renewal-stage customers.
4. Use SHAP and permutation importance to define a reduced-feature challenger. Select features with training and validation data only, retrain the model, and evaluate it once on the untouched test set. Keep the reduced model if it preserves predictive performance while improving interpretability, campaign cost, or operational efficiency.
5. Judge campaign success by churn reduction, retained margin, offer cost, and incremental profit, not recall alone.

## Reproducibility

Run the notebooks from `notebooks/` in this order:

1. `eda.ipynb` creates `clean_data_after_eda.csv`.
2. `feature_engineering.ipynb` creates `my_data_for_prediction.csv`.
3. `modeling.ipynb` trains, evaluates, explains, and persists the models.

Install dependencies with:

```bash
pip install -r requirements.txt
```

The trained models and metadata are saved under `artifacts/`:

- `random_forest_balanced.joblib`
- `xgboost_best.joblib`
- `model_metadata.json`
- `feature_importance_comparison.csv`
- `shap_importance_comparison.csv`

Source data is not included. Place `client_data.csv` and `price_data.csv` under `data/` before running the notebooks.

## Repository structure

```text
.
|-- data/                 # source and generated data files
|-- notebooks/            # EDA, feature engineering, and modeling
|-- artifacts/            # persisted models, metadata, and explainability tables
|-- requirements.txt
`-- README.md
```

## Limitations and next steps

- The analysis is observational and does not prove that price changes cause churn.
- Evaluation uses one stratified split; repeated cross-validation and time-based validation would strengthen the generalization estimate.
- The repository persists trained models but does not yet provide a production inference service.
- The reduced-feature challenger and controlled retention experiment are the next steps for testing whether model explanations improve campaign efficiency.

## Technologies

Python, Jupyter, pandas, NumPy, seaborn, Matplotlib, scikit-learn, XGBoost, SHAP, and joblib.
