"""
This is a boilerplate pipeline 'user_app'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import predict_with_mlflow


def create_pipeline(**kwargs) -> Pipeline:

    pipeline_inference = pipeline(
        [
            node(
                func=predict_with_mlflow,
                inputs=["params:mlflow_inference", "dataset_predition"],
                outputs="prediction",
                name="split_data_node",
                tags=["inference"]
            ),
        ],
    )
    return pipeline_inference
