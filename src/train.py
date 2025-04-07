import argparse
import json
import logging
import os
import sys

import tensorflow as tf
import tensorflow.keras.layers as layers
import yaml
from dvclive import Live
from dvclive.keras import DVCLiveCallback

from custom_callbacks import TrainCustomCallback
from helper_utils import (
    file_data_preprocessing,
    get_dataset,
    get_mask,
    tiled_data_preprocessing,
)
from parameters import (
    IOParameters,
    TrainingParameters,
    model_list_preprocess,
    model_list_size,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,  # Force all logs to stdout
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    # Parse parameters
    io_parameters = IOParameters.parse_obj(parameters["io_parameters"])
    train_parameters = TrainingParameters.parse_obj(parameters["model_parameters"])

    logging.info(tf.test.gpu_device_name())

    # Gather data preprocessing parameters
    batch_size = train_parameters.batch_size
    image_flip = train_parameters.image_flip
    rotation_angle = train_parameters.rotation_angle
    val_pct = train_parameters.val_pct
    seed = train_parameters.seed

    if image_flip.value != "None" and rotation_angle is not None:
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip(image_flip.value, seed=seed),
                layers.RandomRotation(rotation_angle, seed=seed),
            ]
        )
    elif image_flip.value != "None":
        data_augmentation = tf.keras.Sequential(
            [layers.RandomFlip(image_flip.value, seed=seed)]
        )
    elif rotation_angle is not None:
        data_augmentation = tf.keras.Sequential(
            [layers.RandomRotation(rotation_angle, seed=seed)]
        )
    else:
        data_augmentation = tf.keras.Sequential([])

    # Gather training parameters
    weights = train_parameters.weights.value
    epochs = train_parameters.epochs
    nn_model = train_parameters.nn_model.value
    optimizer = train_parameters.optimizer.value
    learning_rate = train_parameters.learning_rate
    loss_func = train_parameters.loss_function.value

    # Prepare data generators and create a tf.data pipeline of augmented images
    dataset, classes, is_tif, dataset_size = get_dataset(
        io_parameters.data_uris,
        io_parameters.root_uri,
        io_parameters.data_type,
        data_tiled_api_key=io_parameters.data_tiled_api_key,
        labels_tiled_uri=io_parameters.labels_tiled_uri,
        labels_tiled_api_key=io_parameters.labels_tiled_api_key,
        shuffle=train_parameters.shuffle,
        seed=seed,
    )

    mask = get_mask(
        mask_tiled_uri=io_parameters.mask_tiled_uri,
        mask_tiled_api_key=io_parameters.mask_tiled_api_key,
    )

    val_size = int(dataset_size * val_pct / 100)
    train_size = dataset_size - val_size
    logging.info(f"Train size: {train_size}, Validation size: {val_size}")

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    target_size = model_list_size[train_parameters.nn_model.name]

    if io_parameters.data_type == "tiled":
        train_generator = train_dataset.map(
            lambda x, y: (
                tiled_data_preprocessing(
                    x,
                    (target_size, target_size),
                    train_parameters.log,
                    train_parameters.low_percentile,
                    train_parameters.high_percentile,
                    mask,
                ),
                y,
            )
        )
        val_generator = val_dataset.map(
            lambda x, y: (
                tiled_data_preprocessing(
                    x,
                    (target_size, target_size),
                    train_parameters.log,
                    train_parameters.low_percentile,
                    train_parameters.high_percentile,
                    mask,
                ),
                y,
            )
        )
    else:
        train_generator = train_dataset.map(
            lambda x, y: (
                file_data_preprocessing(
                    x,
                    (target_size, target_size),
                    is_tif,
                    train_parameters.log,
                    train_parameters.low_percentile,
                    train_parameters.high_percentile,
                    mask,
                ),
                y,
            )
        )
        val_generator = val_dataset.map(
            lambda x, y: (
                file_data_preprocessing(
                    x,
                    (target_size, target_size),
                    is_tif,
                    train_parameters.log,
                    train_parameters.low_percentile,
                    train_parameters.high_percentile,
                    mask,
                ),
                y,
            )
        )

    # Preprocess input according to the neural network
    preprocess_name = model_list_preprocess[train_parameters.nn_model.name]
    preprocess_input = getattr(tf.keras.applications, preprocess_name).preprocess_input

    # Shuffle data at the beginning of each epoch
    if train_parameters.shuffle:
        train_generator = (
            train_generator.shuffle(len(train_generator))
            .batch(batch_size)
            .map(lambda x, y: (preprocess_input(data_augmentation(x)), y))
        )
    else:
        train_generator = train_generator.batch(batch_size).map(
            lambda x, y: (preprocess_input(data_augmentation(x)), y)
        )

    val_generator = val_generator.batch(batch_size).map(
        lambda x, y: (preprocess_input(data_augmentation(x)), y)
    )

    # Define number of classes
    class_num = len(classes)

    # Define optimizer
    opt_code = compile(
        f"tf.keras.optimizers.{optimizer}(learning_rate={learning_rate})",
        "<string>",
        "eval",
    )

    logging.info(f"weights: {weights}")
    if weights != "None":
        # Load pretrained weights
        model_description = f"tf.keras.applications.{nn_model}(include_top=False, \
            input_shape=({target_size},{target_size},3), weights='imagenet', input_tensor=None)"
        model_code = compile(model_description, "<string>", "eval")
        base_model = eval(model_code)
        # Adapt output of the model according to the number of classes
        x = base_model.output
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(4096, activation="relu", name="fc1")(x)
        x = layers.Dense(4096, activation="relu", name="fc2")(x)
        predictions = layers.Dense(class_num, activation="softmax", name="predictions")(
            x
        )
        model = tf.keras.models.Model(
            inputs=base_model.input, outputs=predictions, name=base_model._name
        )
    else:
        model_description = f"tf.keras.applications.{nn_model}(include_top=True, weights=None, \
                             input_tensor=None, classes={class_num})"
        model_code = compile(model_description, "<string>", "eval")
        model = eval(model_code)

    # Compile ML model
    model.compile(optimizer=eval(opt_code), loss=loss_func, metrics=["accuracy"])
    # Print model summary
    model.summary()

    results_dir = f"{io_parameters.results_dir}/{io_parameters.uid_save}"

    live = Live(dir=f"{results_dir}/dvclive", dvcyaml=False)

    # fit model while also keeping track of data for dash plots
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1,
        steps_per_epoch=train_size // batch_size,
        callbacks=[DVCLiveCallback(live=live), TrainCustomCallback(results_dir)],
        shuffle=train_parameters.shuffle,
    )

    # Save model
    model_path = f"{results_dir}/model.keras"
    model.save(model_path)

    # Save class metadata
    class_info_path = f"{results_dir}/class_info.json"
    with open(class_info_path, "w") as json_file:
        json.dump(classes, json_file)

    # Save training history
    history_path = f"{results_dir}/history.json"
    with open(history_path, "w") as json_file:
        json.dump(history.history, json_file)

    # Log artifacts with DVC Live so they are tracked
    # dvclive.log_artifact(model_path)
    # dvclive.log_artifact(class_info_path)
    # dvclive.log_artifact(history_path)

    # # Let dvclive know this training is completed (useful for one-off runs)
    # dvclive.done()

    logging.info("Training process completed")
