import os

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_probability as tfp
from PIL import Image
from tiled.client import from_uri

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
SPLASH_URL = "http://splash:80/api/v0"


def load_from_splash(uri_list, event_id):
    """
    This function loads existing labels from splash-ml
    Args:
        uri_list:       URI of dataset (e.g. file path)
        event_id:       Tagging event id in splash_ml
    Returns:
        labeled_uris:   List of labeled URIs
        labels:         List of assigned labels
    """
    url = f"{SPLASH_URL}/datasets/search"
    params = {"page[offset]": 0, "page[limit]": len(uri_list)}
    data = {"uris": uri_list, "event_id": event_id}
    datasets = requests.post(url, params=params, json=data).json()
    labeled_uris = []
    labels = []
    for dataset in datasets:
        for tag in dataset["tags"]:
            if tag["event_id"] == event_id:
                labels.append(tag["name"])
                labeled_uris.append(dataset["uri"])
    return labeled_uris, labels


def preprocess_image(image, log=False):
    """
    Preprocess image
    Args:
        image:  Image to be preprocessed
        log:    Bool indicating if data should be log transformed
    Returns:
        Image
    """
    if image.dtype != np.uint8:
        # Normalize according to percentiles 1-99
        low = np.percentile(image.ravel(), 1)
        high = np.percentile(image.ravel(), 99)
        image = np.clip((image - low) / (high - low), 0, 1)
        image = (image * 255).astype(np.uint8)  # Convert to uint8, 0-255

    # Check number of channels
    if len(image.shape) == 3:
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.shape[0] == 1:
            image = np.squeeze(image)
        else:
            raise ValueError("Not a valid image shape")

    # Apply log transformation
    if log:
        image = np.log1p(np.array(image))
        image = (
            ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255
        ).astype(np.uint8)

    # Convert to PIL image
    image = Image.fromarray(image)
    image = image.convert("L")
    # image = tf.cast(image, tf.float32) / 255.0
    return image


def gen(root_uri, sub_uris, api_key=None, log=False):
    """
    Generator function to load tiled data
    Args:
        root_uri:       Root URI from which data should be retrieved
        sub_uris:       List of sub URIs
        api_key:        API key for tiled
        log:            Bool indicating if data should be log transformed
    Returns:
        Image tensor
    """
    tiled_client = from_uri(root_uri, api_key=api_key)
    for sub_uri in sub_uris:
        block_array = tiled_client[root_uri][sub_uri]
        for i in range(block_array.shape[0]):
            image = block_array[i,]
            image_tensor = tf.convert_to_tensor(np.array(image))
            yield image_tensor


def get_dataset(data, shuffle=False, event_id=None, seed=42):
    """
    This function prepares the dataset to be used during training and/or prediction processes
    Input:
        data:           Path to parquet file with the list of data
        shuffle:        Bool indicating if the dataset should be shuffled
        event_id:       Tagging event id in splash_ml
        log:            Bool indicating if data should be log transformed
        seed:           Seed for random number generation
    Returns:
        TF dataset, kwargs (classes or filenames), data_type
    """
    # Retrieve data set list
    data_info = pd.read_parquet(data, engine="pyarrow")
    # Retrieve labels
    if event_id:
        # Training
        if data_info["type"][0] == "tiled":
            uri_list = data_info["uri"]
            splash_uri_list = data_info["splash_uri"]
            labeled_uris, labels = load_from_splash(splash_uri_list.tolist(), event_id)
        else:
            uri_list = data_info["uri"]
            labeled_uris, labels = load_from_splash(uri_list.tolist(), event_id)
        if shuffle:
            np.random.permutation(seed)
            indx = np.random.permutation(len(labeled_uris))
            labeled_uris = [labeled_uris[i] for i in indx]
            labels = [labels[i] for i in indx]
        classes = list(set(labels))
        df_labels = pd.DataFrame(labels).replace(
            {class_name: label for label, class_name in enumerate(classes)}
        )
        categorical_labels = tf.keras.utils.to_categorical(
            df_labels.iloc[:, 0].tolist(), num_classes=len(classes)
        )
        dataset = tf.data.Dataset.from_tensor_slices((labeled_uris, categorical_labels))
        kwargs = classes
    else:
        # Inference
        uri_list = data_info["uri"]
        kwargs = uri_list.to_list()
        if data_info["type"][0] == "tiled":
            dataset = tf.data.Dataset.from_generator(
                gen,
                args=(
                    data_info["root_uri"].tolist()[0],
                    data_info["sub_uris"].tolist(),
                    data_info["api_key"].tolist()[0],
                ),
                output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(uri_list)
    # Check if data is in tif format or tiled
    if data_info["type"][0] == "tiled" and not event_id:
        data_type = "tiled"
    elif (
        uri_list[0].split(".")[-1] in ["tif", "tiff", "TIF", "TIFF"]
        or data_info["type"][0] == "tiled"
    ):
        data_type = "tif"
    else:
        data_type = "non-tif"
    return dataset, kwargs, data_type


def data_preprocessing(img, target_shape, data_type, log=False, threshold=1):
    """
    Preprocessing function that loads data per batch
    Args:
        data:           Data to be preprocessed
        target_shape:   Target shape of data
        data_type:      Type of data
        log:            Bool indicating if data should be log transformed
    Returns:
        image
    """
    if data_type != "tiled":
        img = tf.io.read_file(img)
        if data_type != "tif":
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
        else:
            img_tmp = tfio.experimental.image.decode_tiff(img)
            r, g, b = img_tmp[:, :, 0], img_tmp[:, :, 1], img_tmp[:, :, 2]
            img = tf.stack([r, g, b], axis=-1)
    else:
        # Normalize according to percentiles 1-99
        if img.dtype != np.uint8:
            img_flat = tf.reshape(img, [-1])
            low = tfp.stats.percentile(img_flat, 1)
            high = tfp.stats.percentile(img_flat, 99)

            img = tf.clip_by_value((img - low) / (high - low), 0, 1)
            img = tf.cast(img * 255, tf.uint8)

        # Check number of channels
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = tf.transpose(img, (1, 2, 0))
            img = tf.repeat(img, repeats=3, axis=-1)
        elif len(img.shape) == 3 and img.shape[0] == 3:
            img = tf.transpose(img, (1, 2, 0))
        elif len(img.shape) == 2:
            img = tf.expand_dims(img, axis=-1)
            img = tf.repeat(img, repeats=3, axis=-1)
        else:
            raise ValueError("Not a valid image shape")

    img = tf.image.resize(img, tf.constant(target_shape))

    # Apply log transformation
    if log:
        img = tf.math.log(img + threshold)
        img = (
            (img - tf.math.reduce_min(img))
            / (tf.math.reduce_max(img) - tf.math.reduce_min(img))
        ) * 255
        img = tf.cast(img, tf.uint8)
    return img
