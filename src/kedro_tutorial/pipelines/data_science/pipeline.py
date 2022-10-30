"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_data,
    train_model,
    test_predict,
    evaluate_model
)


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_training = pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
                tags=["training"]
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs=["regressor_artifact", "regressor_model"],
                name="train_model_node",
                tags=["training"]
            ),
        ],
    )

    pipeline_evaluate = pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["regressor_artifact", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
                tags=["evaluation", "training"]
            ),
        ]
    )

    pipeline_inference = pipeline(
        [
            node(
                func=test_predict,
                inputs=["X_test", "regressor_artifact"],
                outputs="test_prediction",
                name="test_predict_node",
                tags=["inference"]
            ),
        ]
    )

    return pipeline(
        pipe=(pipeline_training + pipeline_evaluate + pipeline_inference),
        inputs="model_input_table",
        namespace="data_science",
    )
