"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""

import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, max_error


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> [LinearRegression, LinearRegression]:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor, regressor


def test_predict(X_test: pd.DataFrame, model: LinearRegression) -> pd.DataFrame:
    """Outputs predictions

    Args:
        X_test: Dataset for prediction
        model: any regressor

    Returns:
        Prediction on X_test using model with parameters

    """
    return pd.DataFrame(model.predict(X_test), columns=["test_prediction"])


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> [Dict[str, float], Dict[str, float]]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)

    metrics = {"r2_score": {"value": score, "step": 1},
               "mae": {"value": mae, "step": 1},
               "max_error": {"value": me, "step": 1}
               }
    return metrics, metrics
