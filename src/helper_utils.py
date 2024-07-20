import os

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_probability as tfp
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


# def normalize_percentiles(x, low_perc=0.01, high_perc=99):
#     low = np.percentile(x.ravel(), low_perc)
#     high = np.percentile(x.ravel(), high_perc)
#     x = (np.clip((x - low) / (high - low), 0, 1) * 255).astype(np.uint8)
#     return x


def normalize_percentiles(x, low_perc=0.01, high_perc=99, mask=None):
    x = tf.reshape(x, [-1])
    if mask:
        mask = tf.reshape(mask, [-1])
    else:
        # Mask negative and NaN values
        mask_neg = tf.less(x, 0.0)
        mask_nan = tf.math.is_nan(x)
        mask = tf.math.logical_or(mask_neg, mask_nan)

    # Apply the mask
    x = tf.boolean_mask(x, tf.logical_not(mask))

    low = tfp.stats.percentile(x, low_perc)
    high = tfp.stats.percentile(x, high_perc)
    x = tf.clip_by_value((x - low) / (high - low), 0, 1) * 255
    x = tf.cast(x, tf.uint8)
    return x


# def log_transform(image, threshold=0.000000000001):
#     # Mask negative and NaN values
#     nan_img = np.isnan(image)
#     img_neg = (image < 0.0)
#     mask_neg = np.array(img_neg)
#     mask_nan = np.array(nan_img)
#     mask = ((mask_nan + mask_neg))
#     x = np.ma.array(image, mask = mask)

#     image = np.log(x+threshold)
#     x = np.ma.array(image, mask = mask)

#     x = normalize_percentiles(x)
#     return x


def log_transform(image, threshold=1e-12):
    # Mask negative and NaN values
    mask_neg = tf.less(image, 0.0)
    mask_nan = tf.math.is_nan(image)
    mask = tf.math.logical_or(mask_neg, mask_nan)

    # Apply mask
    image = tf.where(mask, 0.0, image)

    # Apply log transform
    image = tf.math.log(image + threshold)

    # Apply mask again
    image = tf.where(mask, 0.0, image)

    # Normalize percentiles
    image = normalize_percentiles(image, mask)

    return image


# def preprocess_image(image, log=False):
#     """
#     Preprocess image
#     Args:
#         image:  Image to be preprocessed
#         log:    Bool indicating if data should be log transformed
#     Returns:
#         Image
#     """

#     # Check number of channels
#     if len(image.shape) == 3:
#         if image.shape[0] == 3:
#             image = np.transpose(image, (1, 2, 0))
#         elif image.shape[0] == 1:
#             image = np.squeeze(image)
#         else:
#             raise ValueError("Not a valid image shape")

#     # Apply log transformation
#     if log:
#         image = log_transform(image)
#     elif image.dtype != np.uint8:
#         image = normalize_percentiles(image)

#     # Convert to PIL image
#     image = Image.fromarray(image)
#     image = image.convert("L")
#     return image


def gen(root_uri, sub_uris, api_key=None):
    """
    Generator function to load tiled data
    Args:
        root_uri:       Root URI from which data should be retrieved
        sub_uris:       List of sub URIs
        api_key:        API key for tiled
    Returns:
        Image tensor
    """
    tiled_client = from_uri(root_uri.decode("ascii"), api_key=api_key)
    for sub_uri in sub_uris:
        block_array = tiled_client[sub_uri.decode("ascii")]
        if len(block_array.shape) > 2:
            for i in range(block_array.shape[0]):
                # image = block_array[i,]
                # image_tensor = tf.convert_to_tensor(np.array(image))
                yield block_array[i,]
        else:
            yield block_array[:]


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
            _, labels = load_from_splash(splash_uri_list.tolist(), event_id)
            labeled_uris = uri_list.tolist()

        else:
            uri_list = data_info["uri"]

            # Check if path correction was performed due to running outside containers
            if "splash_uri" in data_info.columns:
                splash_uri_list = data_info["splash_uri"].tolist()
                labeled_splash_uri_list, labels = load_from_splash(
                    splash_uri_list, event_id
                )

                labeled_indices = []
                for item in labeled_splash_uri_list:
                    if item in splash_uri_list:
                        labeled_indices.append(splash_uri_list.index(item))

                labeled_uris = [uri_list.tolist()[i] for i in labeled_indices]

            else:
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
        if data_info["type"][0] == "tiled":

            tiled_root_uri = data_info["root_uri"].tolist()[0]
            tiled_sub_uris = data_info["sub_uris"].tolist()
            tiled_key = data_info["api_key"].tolist()[0]

            if tiled_key:
                dataset = tf.data.Dataset.from_generator(
                    gen,
                    args=(
                        tiled_root_uri,
                        tiled_sub_uris,
                        tiled_key,
                    ),
                    output_signature=tf.TensorSpec(
                        shape=(None, None), dtype=tf.float32
                    ),
                )
            else:
                dataset = tf.data.Dataset.from_generator(
                    gen,
                    args=(
                        tiled_root_uri,
                        tiled_sub_uris,
                    ),
                    output_signature=tf.TensorSpec(
                        shape=(None, None), dtype=tf.float32
                    ),
                )
            kwargs = None

        else:
            uri_list = data_info["uri"]
            dataset = tf.data.Dataset.from_tensor_slices(uri_list)
            kwargs = uri_list.to_list()

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

        if log:
            img = log_transform(img)

    else:
        # Check number of channels
        if len(img.shape) == 3:
            if img.shape[0] == 3:
                img = tf.transpose(img, perm=[1, 2, 0])
            elif img.shape[0] == 1:
                img = tf.squeeze(img)
            else:
                raise ValueError("Not a valid image shape")

        if log:
            img = log_transform(img)

        # Convert to RGB
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

    return img
