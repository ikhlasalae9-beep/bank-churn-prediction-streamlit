# Bank Customer Churn Prediction Streamlit App

Professional Streamlit dashboard for a saved Bank Customer Churn Prediction scikit-learn pipeline.

The app does not retrain models. It loads the saved pipeline and reproduces the feature engineering contract used in the notebook.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run App

```bash
streamlit run app.py
```

## Required Project Files

Preferred model path:

```text
saved_models/final_churn_model.pkl
```

Fallback model path:

```text
saved_models/best_churn_model.pkl
```

Preferred dataset path:

```text
data/Churn_Modelling.csv
```

The app also supports the dataset at the project root as `Churn_Modelling.csv`.

## Batch Prediction CSV Columns

Uploaded CSV files must contain these columns:

```text
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
```

Extra columns are ignored for prediction. `HasCrCard` and `IsActiveMember` may be provided as `Yes`/`No`, `True`/`False`, or `1`/`0`.

## Pages

- Home
- Single Customer Prediction
- Batch CSV Prediction
- Prediction History
- Exploratory Data Analysis
- Model Performance
- Model Information

## Prediction History

The app stores successful single and batch predictions locally at:

```text
data/prediction_history.csv
```

If the file does not exist, the app creates it automatically. The Prediction History page lets you filter, download, and clear the saved history.

## Common Error

If you get a missing columns error such as:

```text
columns are missing: {'OneProduct', 'TwoProducts'}
```

it means the feature engineering function in `app.py` is not synchronized with the notebook. The saved pipeline uses `FunctionTransformer(add_churn_features)`, so `add_churn_features` must exist before `joblib.load(...)` and must create every engineered feature expected by the pipeline.
