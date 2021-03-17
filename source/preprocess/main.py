import sys
import numpy as np
from sklearn.model_selection import train_test_split

from source.utils import helper
import source.argsparser as p


def do_train_test_split(
        path: str,
        label: str,
        env: str,
        test_size: float = 0.20,
        val_size: float = 0.20
):
    X = [
        i.name for i in dbutils.fs.ls(
            path +
            "/" +
            label) if label and "jpg" in i.name]
    Y = np.zeros(len(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=7
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        Y,
        test_size=val_size,
        random_state=7
    )

    print("-" * 50)
    print(f"Label: {label}")
    print(f"Total images: {len(X)}")
    print(f"Train images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")
    print(f"Test images: {len(X_test)}")
    return X_train, X_val, X_test


def prep_write_img(
        image: str,
        path: str,
        env: str,
        source: str,
        label: str,
        resize: False,
        height: int = 200,
        width: int = 200
):
    import re
    import os
    import cv2
    path = re.sub(r"dbfs:", "/dbfs", path)
    try:
        # Load + Prep image
        img = cv2.imread(os.path.join(path, label, image))
        if resize:
            img = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)
        # Write Image
        cv2.imwrite(
            os.path.join(
                path,
                "Preprocessed",
                source,
                label,
                image),
            img)
    except Exception as e:
        print(f"Image Resize Error: --> {image}: {str(e)}")
        pass
    return image


def main(path: str, env: str, label: str):
    """
    Process Train/Validation Images

    Keyword arguments
    ----------
    path: str
        input data path
    env: str
        environment
    label: str
        image class label name
    """
    print(f"\nProcessing {label} images.....")
    X_train, X_val, X_test = do_train_test_split(path, label, env)

    # Prep images
    for image, source in zip(
        [X_train, X_val, X_test],
            ["Train", "Validation", "Test"]):
        print(f"processing {source}....")
        sc.parallelize(image).map(
            lambda img: prep_write_img(
                img,
                path,
                env,
                source,
                label)).collect()
        print("Done!")


if __name__ == "__main__":

    # Input Arguments
    parser = p.parser([sys.argv[1:]])
    args = parser.parse_args()

    if args.dbconnect:
        spark, dbutils = helper.config_spark_dbutils()
        sc = spark.sparkContext

        # Train: Load + Process
        path = f"dbfs:/mnt/{args.env}/images/PetImages"
        for label in ["Cat", "Dog"]:
            main(path, args.env, label)
