import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.keras.models import load_model

from custom_callbacks import PredictionCustomCallback
from helper_utils import (
    file_data_preprocessing,
    get_dataset,
    get_mask,
    tiled_data_preprocessing,
    write_results,
)
from parameters import (
    InferenceParameters,
    IOParameters,
    model_list_preprocess,
    model_list_size,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,  # Force all logs to stdout
)
logger = logging.getLogger(__name__)


# Prepare the trained model to return both the probabilities and feature vectors
class CustomModel(tf.keras.Model):
    def __init__(self, trained_model):
        super(CustomModel, self).__init__()
        self.base_model = tf.keras.Model(
            inputs=trained_model.input, outputs=trained_model.layers[-2].output
        )
        self.last_layers = tf.keras.Model(
            inputs=trained_model.layers[-2].output, outputs=trained_model.output
        )

    def call(self, inputs, training=False):
        second_to_last = self.base_model(inputs)
        x = self.last_layers(second_to_last)
        return x, second_to_last


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    # Parse parameters
    io_parameters = IOParameters.parse_obj(parameters["io_parameters"])
    inference_parameters = InferenceParameters.parse_obj(parameters["model_parameters"])

    logging.info(tf.test.gpu_device_name())

    # Load trained model and parameters
    models_dir = io_parameters.models_dir
    model_dir = f"{models_dir}/{io_parameters.uid_retrieve}"
    loaded_model = load_model(model_dir + "/model.keras")
    target_size = model_list_size[loaded_model._name]
    custom_model = CustomModel(
        loaded_model
    )  # Modify trained model to return prob and f_vec

    # Prepare data generators and create a tf.data pipeline of augmented images
    predict_dataset, classes, is_tif = get_dataset(
        io_parameters.data_uris,
        io_parameters.root_uri,
        io_parameters.data_type,
        data_tiled_api_key=io_parameters.data_tiled_api_key,
        shuffle=False,
    )

    mask = get_mask(
        mask_tiled_uri=io_parameters.mask_tiled_uri,
        mask_tiled_api_key=io_parameters.mask_tiled_api_key,
    )

    if io_parameters.data_type == "tiled":
        predict_generator = predict_dataset.map(
            lambda x: (
                tiled_data_preprocessing(
                    x,
                    (target_size, target_size),
                    inference_parameters.log,
                    inference_parameters.low_percentile,
                    inference_parameters.high_percentile,
                    mask,
                )
            )
        )
    else:
        predict_generator = predict_dataset.map(
            lambda x: (
                file_data_preprocessing(
                    x,
                    (target_size, target_size),
                    is_tif,
                    inference_parameters.log,
                    inference_parameters.low_percentile,
                    inference_parameters.high_percentile,
                    mask,
                )
            )
        )

    # Preprocess input according to the model if weights are set to imagenet
    batch_size = inference_parameters.batch_size
    preprocess_name = model_list_preprocess[loaded_model._name]
    preprocess_input = getattr(tf.keras.applications, preprocess_name).preprocess_input
    predict_generator = predict_generator.batch(batch_size).map(
        lambda x: (preprocess_input(x))
    )

    with open(model_dir + "/class_info.json", "r") as json_file:
        classes = json.load(json_file)
    class_num = len(classes)

    # Start prediction process
    prob, f_vec = custom_model.predict(
        predict_generator,
        verbose=0,
        callbacks=[PredictionCustomCallback(classes=classes)],
    )

    df_results = pd.DataFrame(prob, columns=classes)

    df_f_vec = pd.DataFrame(f_vec)
    df_f_vec.columns = df_f_vec.columns.astype(str)

    # Create output directory if it does not exist
    output_dir = f"{io_parameters.results_dir}/{io_parameters.uid_save}"
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(f"{output_dir}/results.parquet", engine="pyarrow")
    df_f_vec.to_parquet(f"{output_dir}/f_vectors.parquet", engine="pyarrow")
    logger.info(f"Results written to {output_dir}")

    # Write results to Tiled
    write_results(
        df_f_vec,
        df_results,
        io_parameters,
        f"{output_dir}/f_vectors.parquet",
        f"{output_dir}/results.parquet",
        parameters,
    )
    logger.info("Results written to tiled")
