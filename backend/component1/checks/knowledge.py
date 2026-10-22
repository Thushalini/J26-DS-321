"""What usually goes with what in a data science plan. Names match extraction/rules.py.

Kept deliberately small and conventional, so every rule can be defended from standard practice:
regression metrics measure distances between numbers, classification metrics count correct
labels, clustering has no labels at all, and forecasting is regression over time.
"""
from __future__ import annotations

TASK_METRICS: dict[str, set[str]] = {
    "classification": {"Accuracy", "Precision", "Recall", "F1-score", "ROC AUC", "Log Loss", "Cohen's Kappa"},
    "regression": {"RMSE", "MSE", "MAE", "MAPE", "R-squared"},
    "forecasting": {"RMSE", "MSE", "MAE", "MAPE", "R-squared"},
    "clustering": {"Silhouette Score"},
    "anomaly detection": {"Precision", "Recall", "F1-score", "ROC AUC"},
    "recommendation": {"Precision", "Recall"},
}
METRIC_ALTERNATIVES: dict[str, list[str]] = {
    "classification": ["F1-score", "ROC AUC"],
    "regression": ["RMSE", "MAE"],
    "forecasting": ["MAPE", "RMSE"],
    "clustering": ["Silhouette Score"],
    "anomaly detection": ["F1-score", "ROC AUC"],
    "recommendation": ["Precision", "Recall"],
}

_TREES = {"XGBoost", "LightGBM", "CatBoost", "Random Forest", "Decision Tree", "Gradient Boosting", "AdaBoost"}
_DEEP = {"CNN", "RNN", "LSTM", "GRU", "Transformer", "BERT", "Neural Network"}
TASK_ALGORITHMS: dict[str, set[str]] = {
    "classification": _TREES | _DEEP | {"Logistic Regression", "SVM", "KNN", "Naive Bayes"},
    "regression": _TREES | _DEEP | {"Linear Regression", "Ridge Regression", "Lasso Regression", "SVM", "KNN"},
    "forecasting": _TREES | {"ARIMA", "SARIMA", "Prophet", "LSTM", "GRU", "RNN", "Transformer", "Neural Network",
                             "Linear Regression"},
    "clustering": {"K-Means", "DBSCAN"},
    "anomaly detection": _TREES | {"DBSCAN", "K-Means", "SVM", "KNN", "Neural Network", "LSTM"},
    "recommendation": {"KNN", "Neural Network", "Transformer", "XGBoost"},
}
ALGORITHM_ALTERNATIVES: dict[str, list[str]] = {
    "classification": ["Random Forest", "XGBoost"],
    "regression": ["XGBoost", "Linear Regression"],
    "forecasting": ["ARIMA", "Prophet"],
    "clustering": ["K-Means", "DBSCAN"],
    "anomaly detection": ["DBSCAN", "Random Forest"],
    "recommendation": ["KNN"],
}
# What kind of method a name is, for the explanation ("K-Means is a clustering method").
ALGORITHM_FAMILY: dict[str, str] = {
    "K-Means": "a clustering method", "DBSCAN": "a clustering method",
    "ARIMA": "a time-series forecasting method", "SARIMA": "a time-series forecasting method",
    "Prophet": "a time-series forecasting method",
    "Logistic Regression": "a classification method", "Naive Bayes": "a classification method",
    "Linear Regression": "a regression method", "Ridge Regression": "a regression method",
    "Lasso Regression": "a regression method",
}
METRIC_FAMILY: dict[str, str] = {
    **{m: "a regression metric" for m in ("RMSE", "MSE", "MAE", "MAPE", "R-squared")},
    **{m: "a classification metric" for m in ("Accuracy", "Precision", "Recall", "F1-score", "ROC AUC", "Log Loss",
                                              "Cohen's Kappa")},
    "Silhouette Score": "a clustering metric",
}

DEEP_LEARNING = _DEEP
COMPUTE_RESOURCES = {"GPU", "TPU", "Google Colab", "Kaggle notebooks", "AWS", "Azure", "Google Cloud"}
SIMPLER_BASELINES: dict[str, list[str]] = {
    "classification": ["XGBoost", "Logistic Regression"],
    "regression": ["XGBoost", "Linear Regression"],
    "forecasting": ["ARIMA", "Prophet"],
}
SHORT_TIMELINE_DAYS = 28          # under four weeks left is little time to train and tune a deep model

# Words that make a target look like a number or like a category. A hint, never a certainty,
# so a mismatch is only ever a warning.
NUMBER_TARGET_WORDS = ("arrivals", "price", "prices", "sales", "revenue", "demand", "count", "number of",
                       "amount", "cost", "income", "temperature", "duration", "volume", "rate", "quantity",
                       "visitors", "occupancy", "load", "consumption", "rainfall", "yield")
CATEGORY_TARGET_WORDS = ("churn", "fraud", "default", "spam", "class", "category", "type", "label", "whether",
                         "yes/no", "diagnosis", "disease", "sentiment", "survival", "approval", "pass/fail",
                         "risk level", "attrition")
NUMBER_TASKS = ("regression", "forecasting")
CATEGORY_TASKS = ("classification",)
