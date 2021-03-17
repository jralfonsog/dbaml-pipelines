import sys
import json
import requests
import numpy as np

from source.utils import helper
import source.argsparser as p


def predict(scoring_uri: str, image: np.array):
    """ Request prediction from Azure deployed webservice

    Args:
        scoring_uri: webservice scoring uri
        image: input image array
    """
    sample_image = {"data": image.tolist()}
    response = requests.post(
        url=scoring_uri,
        data=json.dumps(sample_image),
        headers={"Content-type": "application/json"}
    )
    response = json.loads(response.content)
    label = "Cat" if response == 0 else "Dog"
    print(f"Prediction: {label}")


if __name__ == "__main__":

    # Input arguments
    parser = p.parser([sys.argv[1:]])
    args = parser.parse_args()

    if args.dbconnect:
        spark, dbutils = helper.config_spark_dbutils()
        sc = spark.sparkContext
