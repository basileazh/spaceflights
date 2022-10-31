"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, max_error

# from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor



# from tpot import TPOTRegressor
# from tpot.export_utils import set_param_recursive

import logging
logger = logging.getLogger(__name__)


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    X = X.replace({True: 1, False: 0})
    y = data["price"]
    X_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return pd.DataFrame(X_train), pd.DataFrame(x_test), pd.DataFrame(y_train), pd.DataFrame(y_test)


def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict):
    """Trains the linear regression model.

    Args:
        parameters: Parameters defined in parameters/data_science.yml.
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """

    logger.info(f"x_train : {x_train.shape}")
    logger.info(f"y_train : {y_train.shape}")

    # xgb
    regressor = GradientBoostingRegressor()
    logger.info(f"GradientBoostingRegressor instanciated")
    x_train = x_train.astype(float).to_numpy().astype(np.float64)
    y_train = y_train.astype(float).to_numpy().astype(np.float64)
    regressor.fit(x_train, y_train)
    logger.info(f"Model trained")
    # pipelines = [
    #     [
    #         parameters['pipeline'][i],
    #         eval(f"{parameters['pipeline'][i]}()")
    #     ]
    #     for i in range(len(parameters['pipeline']))
    # ]
    # logger.info(pipelines)
    #
    # model_name = []
    # results = []
    # for pipe, model in pipelines:
    #     crossv_results = cross_val_score(model, **parameters['cross_val_score'])
    #     results.append(crossv_results)
    #     model_name.append(pipe)
    #     msg = "%s: %f (%f)" % (model_name, crossv_results.mean(), crossv_results.std())
    #     print(msg)

    return regressor, regressor


def test_predict(x_test: pd.DataFrame, model) -> pd.DataFrame:
    """Outputs predictions

    Args:
        x_test: Dataset for prediction
        model: any regressor

    Returns:
        Prediction on x_test using model with parameters

    """
    return pd.DataFrame(model.predict(x_test), columns=["test_prediction"])


def evaluate_model(regressor, x_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        x_test: Testing data of independent features.
        y_test: Testing data for price.

    Returns:
        dict of metrics, as following : {"r2_score": {"value": score, "step": 1},
               "mae": {"value": mae, "step": 1},
               "max_error": {"value": me, "step": 1}
               }

    """

    y_pred = regressor.predict(x_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)

    metrics = {
        "r2_score": {"value": score, "step": 1},
        "mae": {"value": mae, "step": 1},
        "max_error": {"value": me, "step": 1}
    }
    return metrics
