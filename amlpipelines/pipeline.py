import argparse
from typing import List

from azureml.core import Experiment, Workspace
from azureml.core.compute import DatabricksCompute
from azureml.pipeline.core import Pipeline, PublishedPipeline, ScheduleRecurrence
from azureml.pipeline.steps import DatabricksStep

from amlpipelines.tools import utils, config


class BaseDBStage:
    """ Base pipeline builder for other pipelines to inherit from.
    Specifies naming standards for pipelines and helpers for organizing work.

    Args:
        workspace: AzureML workspace
        cluster_params: Dict of cluster parameters from utils.get_cluster_params
        environment: Databricks environment
    """
    def __init__(
            self,
            workspace: Workspace,
            cluster_params: dict,
            environment: str,
            **kwargs
    ):
        self.ws = workspace
        self.cluster_params = cluster_params
        self.environment = environment

    def __repr__(self) -> str:
        """ String representation of class."""
        return f"PipelineWrapper(workspace='{self.ws.name}', " \
               f"rg='{self.ws.resource_group}', " \
               f"location='{self.ws.location}', " \
               f"subscription={self.ws.subscription_id})"

    def _db_step(
            self,
            name: str,
            python_script_name: str,
            python_script_params: List[str] = []
    ) -> DatabricksStep:
        """ Creates a new DatabricksStep object for use within
        pipelines. Reduces redundancy in calls

        Args:
            name: Step name
            python_script_name: Script to call
            python_script_params: Additional parameters to send to script
        """
        return DatabricksStep(
            name=name,
            source_directory=".",
            python_script_name=python_script_name,
            python_script_params=(["--env", self.environment] + python_script_params),
            compute_target=DatabricksCompute(
                workspace=self.ws,
                name=self.cluster_params["aml_attached"]),
            allow_reuse=True,
            existing_cluster_id=self.cluster_params["cluster_id"]
            )

    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        pass

    def add(self, steps: List[DatabricksStep] = []) -> List[DatabricksStep]:
        """Add PipelineSteps corresponding to what is needed in the AML pipeline."""
        raise NotImplementedError


def submit_pipeline(
        workspace: Workspace,
        experiment: Experiment,
        pipeline: Pipeline,
        show_output: bool = True
) -> Pipeline:
    """Submit the pipeline to AML, which runs but does not save the pipeline

    Args:
        workspace:
        experiment:
        pipeline:
        show_output: If true, shows output from submitting pipeline
    """
    submit = experiment.submit(pipeline, regenerate_outputs=False)
    submit.wait_for_completion(show_output=show_output)
    return submit


def publish_pipeline(
        workspace: Workspace,
        experiment: Experiment,
        pipeline: Pipeline,
        name: str,
        description: str,
        delete_existing: bool = True,
        schedule: ScheduleRecurrence = None
) -> PublishedPipeline:
    """Publish a pipeline to AML, which runs and then saves the pipeline

    Return a PublishedPipeline object which may be used for later scheduling

    Args:
        workspace:
        experiment:
        pipeline:
        name:
        description:
        delete_existing: Delete any existing pipelines/schedules with the same name (default True)
        schedule: Create a prespecified schedule for the pipeline (default True)
    """
    published_pipeline = pipeline.publish(name=name, description=description)
    if schedule is not None:
        utils.schedule_recurring(workspace, published_pipeline.id, experiment.name, schedule)
    return published_pipeline


def create_and_run_pipeline(
        stages: List[BaseDBStage],
        experiment_name: str,
        name: str,
        description: str,
        action: str,
        schedule: ScheduleRecurrence = None
) -> Pipeline:
    # build the argsparser
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--remote", action="store_true", default=False)
    parser.add_argument("--env", type=str, choices=["prod", "dev"], default="dev")
    
    if action == "query":
        parser.add_argument(
            "--action",
            type=str,
            choices=["submit", "publish"],
            help="'submit' (run and do not reuse) or 'publish' a pipeline",
            default="publish"
        )
    for stage_cls in stages:
        stage_cls.add_cli_args(parser)

    args = parser.parse_args()
    workspace = utils.get_aml_workspace(args.remote, args.env)
    experiment = Experiment(workspace, experiment_name)
    cluster_params = config.get_cluster_params(workspace, args.env)
    if action == "query":
        action = args.action

    # create pipeline steps
    args = vars(args)
    steps = []
    for stage_cls in stages:
        db_step = stage_cls(
            environment=args["env"],
            cluster_params=cluster_params,
            workspace=workspace,
            **args
        )
        steps = db_step.add(steps)

    # create the pipeline
    pipeline = Pipeline(
        workspace=workspace,
        steps=steps,
        description=description
    )
    pipeline.validate()

    if action == "submit":
        return submit_pipeline(workspace, experiment, pipeline, show_output=True)

    elif action == "publish":
        return publish_pipeline(
            workspace,
            experiment,
            pipeline,
            name=name,
            description=description,
            delete_existing=True,
            schedule=schedule
        )
