"""
This is a boilerplate pipeline 'user_app'
generated using Kedro 0.18.3
"""
import mlflow
import mlflow.pyfunc
import pandas as pd


# def predict_with_mlflow(parameters, dataset_predition: pd.DataFrame) -> pd.DataFrame:
def predict_with_mlflow(parameters, dataset_predition: pd.DataFrame) -> pd.DataFrame:

    # logged_model = f'runs:/{parameters["run_id"]}/spaceflight'
    # # Load model as a PyFuncModel.
    # loaded_model = mlflow.pyfunc.load_model(logged_model)
    # # Predict on a Pandas DataFrame.
    # prediction = loaded_model.predict(dataset_predition)
    # return prediction

    model_name = parameters["model_name"]
    stage = parameters["stage"]

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )

    return model.predict(dataset_predition)

