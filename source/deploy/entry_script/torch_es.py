import json
import torch
import mlflow
import numpy as np
import mlflow.pytorch
from torchvision import transforms
from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path(
        model_name="torch-catdog-model",
        version=None
    )
    model = mlflow.pytorch.load_model(model_path)


def run(json_input):
    try:
        loader = transforms.Compose([transforms.ToTensor()])
        image = np.array(json.loads(json_input)["data"])
        image = loader(image).float()
        image = image.unsqueeze(0)
        output = model(image)
        prediction = int(torch.max(output.data, 1)[1].numpy())
        return prediction
    except Exception as e:
        error = str(e)
        return error
