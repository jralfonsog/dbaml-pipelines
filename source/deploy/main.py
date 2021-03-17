import os
import sys
import azureml
import numpy as np

import torch
import torchvision

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking.client import MlflowClient

import azureml.core
from azureml.core.model import InferenceConfig
from azureml.core import Workspace, Model
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.authentication import ServicePrincipalAuthentication

import source.argsparser as p
from source.utils import helper


def get_best_model(exp_name: str, metric: str, tracking_uri="databricks"):
    """ Get best model from MLflow

    Args:
        exp_name: MLflow experiment name
        metric: scoring metric to filter best model
        tracking_uri: remote tracking uri (usually databricks://user)
    """

    mlflow.set_tracking_uri(tracking_uri)
    print("MLflow tracking URI: ", mlflow.get_tracking_uri())

    expid = MlflowClient().get_experiment_by_name(exp_name).experiment_id
    runid = MlflowClient().search_runs(
        experiment_ids=expid,
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=[f"metrics.{metric} DESC"]
    )[0].info.run_id

    print(f"MLflow exp: {exp_name} -- Exp ID: {expid} -- Best Model: {runid}")
    return expid, runid


def register_model(
        ws: Workspace,
        model_name: str,
        model_path: str = "models/"
):
    """ Register model to Azure Model Registry
    
    Args:
        ws: AzureML workspace
        model_name: name of the model to save as in Azure
        model_path: path of mlflow model artifact
    """
    try:
        Model.register(
            model_path=model_path,
            model_name=model_name,
            workspace=ws
        )
        print("Model successfully registered")
    except Exception as e:
        print(f"Failed to register model: -- {str(e)}")


def deploy_model(
        ws: Workspace,
        model_name: str,
        deploy_name: str,
        inference_path: str,
        n_cores: int = 2,
        memory_gb: int = 4
):
    """ Deploy model as a webservice

    Args:
        ws: AzureML workspace
        model_name: name of registered model for deployment
        deploy_name: name of deployed webservice
        inference_path: relative path for inference entry script and env config
        n_cores: number of cpus for webservice
        memory_gb: allocated memory for webservice
    """
    try:
        Webservice(ws, deploy_name).delete()
    except:
        print(f"{deploy_name} Webservice not fount...")
        pass

    # !! Conda environment --> default configured for pytorch
    env = Environment(name="inference-env")
    cd = CondaDependencies().create(python_versions="3.7.7")
    pkg_list = [
        f"numpy=={np.__version__}",
        f"torch=={torch.__version__}",
        f"torchvision=={torchvision.__version__}",
        f"mlflow=={mlflow.__version__}",
        f"azureml-core=={azureml.core.VERSION}"
    ]
    for pkg in pkg_list:
        cd.add_pip_package(pkg)
    env.python.conda_dependencies = cd

    # Configure inference image
    aci_config = AciWebservice.deploy_configuration(
        cpu_cores=n_cores,
        memory_gb=memory_gb
    )
    inference_config = InferenceConfig(
        entry_script=inference_path,
        environment=env
    )

    print(f"Deploying {deploy_name} webservice...")
    webservice = Model.deploy(
        workspace=ws,
        name=deploy_name,
        deployment_config=aci_config,
        inference_config=inference_config,
        models=[Model(ws, name=model_name)]
    )

    webservice.wait_for_deployment(show_output=True)
    print("Done!")
    return webservice


if __name__ == "__main__":

    # Input arguments
    parser = p.parser([sys.argv[1:]])
    args = parser.parse_args()
