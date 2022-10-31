"""
This is a boilerplate pipeline 'user_app'
generated using Kedro 0.18.3
"""
import mlflow
import pandas as pd


def predict_with_mlflow(parameters, dataset_predition: pd.DataFrame) -> pd.DataFrame:

    logged_model = f'runs:/{parameters["run_id"]}/spaceflight'
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    # Predict on a Pandas DataFrame.
    return loaded_model.predict(dataset_predition)
