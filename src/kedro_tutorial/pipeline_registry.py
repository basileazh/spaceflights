"""Project pipelines."""
from typing import Dict
from platform import python_version

from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline.pipeline_ml_factory import pipeline_ml_factory

from kedro_tutorial import __version__ as PROJECT_VERSION
from kedro_tutorial.pipelines import data_processing as dp
from kedro_tutorial.pipelines import data_science as ds
from kedro_tutorial.pipelines import user_app as ua


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    # Data Processing
    data_processing_pipeline = dp.create_pipeline()

    # Data Science
    data_science_pipeline = ds.create_pipeline()

    inference_pipeline = data_science_pipeline.only_nodes_with_tags("inference")
    evaluation_pipeline = data_science_pipeline.only_nodes_with_tags("evaluation")
    training_pipeline = pipeline_ml_factory(
        training=data_science_pipeline.only_nodes_with_tags("training"),
        inference=inference_pipeline,
        input_name="data_science.X_test",
        log_model_kwargs=dict(
            artifact_path="spaceflight",
            conda_env={
                "python": python_version(),
                "build_dependencies": ["pip"],
                "dependencies": [f"kedro_tutorial == {PROJECT_VERSION}"],
            },
            signature="auto",
        ),
    )

    user_app_pipeline = ua.create_pipeline()

    return {
        "__default__": user_app_pipeline,
        "dp": data_processing_pipeline,
        "training": training_pipeline,
        "inference": inference_pipeline,
        "evaluation": evaluation_pipeline,
        "user_app": user_app_pipeline,
    }
