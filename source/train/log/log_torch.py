import mlflow
import mlflow.pytorch
from typing import List


def do_mlflow(metrics: List[str], exp_name: str) -> None:
    """MLflow: Log model metrics + Save models

    Args:
        metrics: input list of metrics recorded from training
        exp_name: experiment name
    """

    # Model + Metrics
    history, acc, params, model = metrics

    train_acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    train_loss = history["loss"]
    val_loss = history["val_loss"]

    # Set MLflow Tracking
    try:
        mlflow.create_experiment(exp_name)
        print("Creating MLflow experiment: ", exp_name)
    except BaseException:
        print("MLflow experiment already exists! -- Using: ", exp_name)
        pass

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(exp_name)
    print(f"** MLflow experiment: {exp_name} **")
    
    with mlflow.start_run():
        for p in [
            "model_type",
            "img_size",
            "augment",
            "optimizer",
            "dropout",
            "dense_units",
            "bath_norm"
        ]:
            mlflow.log_param(p, params[p])
        mlflow.log_param("framework", "pytorch")

        # Log metrics per epoch
        for idx, epoch in enumerate(range(1, len(train_acc) + 1)):
            mlflow.log_metric("train_acc", train_acc[idx], step=epoch)
            mlflow.log_metric("val_acc", val_acc[idx], step=epoch)
            mlflow.log_metric("train_loss", train_loss[idx], step=epoch)
            mlflow.log_metric("val_loss", val_loss[idx], step=epoch)

        mlflow.log_metric("test_acc", acc)
        mlflow.pytorch.log_model(model, "models")
    return None
