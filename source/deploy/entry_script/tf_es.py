import json
import mlflow
import mlflow.keras
import numpy as np
from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path(
        model_name="tf-catdog-model",
        version=None
    )
    model = mlflow.keras.load_model(model_path)


def run(json_input):
    try:
        image = np.array(json.loads(json_input)["data"])
        pred = model.predict(image)
        pred = np.argmax(pred, axis=1)
        return pred.tolist()
    except Exception as e:
        error = str(e)
        return error
