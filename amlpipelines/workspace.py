import argparse
import logging
import os

from azureml.core import Workspace


def config_aml_workspace(
        environment: str,
        workspace_prefix: str,
        subscription_id: str,
        resource_group: str,
        storage_account_name: str,
        keyvault_name: str,
        location: str
) -> Workspace:

    """

    Args:
        environment: dev, test, prod
        workspace_prefix: Name prefix for AML workspaces
        subscription_id: Subscription ID
        resource_group: Resource group of items
        storage_account_name: Blob storage account connected to AML
        keyvault_name: Secret storage account
        location: Location of all resources

    Returns:
        Workspace: Azure Workspace

    """

    workspace_name = f"{workspace_prefix}-{environment}"

    id_prefix = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers"
    key_vault = f"{id_prefix}/microsoft.keyvault/vaults/{keyvault_name}"
    storage_account = f"id_prefix/microsoft.storage/storageaccounts/{storage_account_name}{environment}"

    try:
        ws = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group
        )
        logging.info(f"Workspace {workspace_name} already exists, skipping creation.")
    except BaseException:
        logging.info(f"Creating new workspace: {workspace_name}")
        ws = Workspace.create(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            location=location,
            key_vault=key_vault,
            storage_account=storage_account
        )
        ws.get_details()
        ws.write_config(
            path="amlpipelines/configs",
            file_name=f"config-{environment}"
        )

    return ws
