import glob
import os
from functools import partial, reduce

import imageio.v3 as iio
import numpy as np
import tensorflow as tf

# import tensorflow_io as tfio
import tensorflow_probability as tfp
from tiled.client import from_uri
from tiled.structures.data_source import Asset, DataSource
from tiled.structures.table import TableStructure

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

FORMATS = [
    "*.[pP][nN][gG]",
    "*.[jJ][pP][gG]",
    "*.[jJ][pP][eE][gG]",
    "*.[tT][iI][fF]",
    "*.[tT][iI][fF][fF]",
]

NOT_ALLOWED_FORMATS = [
    "**/__pycache__/**",
    "**/.*",
    "cache/",
    "cache/**/",
    "cache/**",
    "tiled_local_copy/",
    "**/tiled_local_copy/**",
    "**/tiled_local_copy/**/",
    "mlexchange_store/**/",
    "mlexchange_store/**",
    "labelmaker_outputs/**/",
    "labelmaker_outputs/**",
]


def filepaths_from_directory(
    root_uri, selected_sub_uris=None, formats=FORMATS, sort=True
):
    """
    This function returns the list of file paths from the directory
    Args:
        root_uri:           [str] Root URI
        selected_sub_uris:  [list] List of selected sub URIs
    Returns:
        List of file paths
    """
    filenames = []
    for dataset in selected_sub_uris:
        dataset_path = os.path.join(root_uri, dataset)
        if os.path.isdir(dataset_path):
            # Find paths that match the format of interest
            all_paths = list(
                reduce(
                    lambda list1, list2: list1 + list2,
                    (
                        [
                            path
                            for path in glob.glob(
                                str(dataset_path) + "/" + t, recursive=False
                            )
                        ]
                        for t in formats
                    ),
                )
            )
            # Find paths that match the not allowed file/directory formats
            not_allowed_paths = list(
                reduce(
                    lambda list1, list2: list1 + list2,
                    (
                        [
                            path
                            for path in glob.glob(
                                str(dataset_path) + "/" + t, recursive=False
                            )
                        ]
                        for t in NOT_ALLOWED_FORMATS
                    ),
                )
            )
            # Remove not allowed filepaths from filepaths of interest
            paths = list(set(all_paths) - set(not_allowed_paths))
            if sort:
                paths.sort()
            filenames += paths
    return filenames


def load_labeled_data(
    tiled_uri,
    tiled_api_key=None,
):
    """
    This function loads existing labels from tiled.

    Args:
        tiled_uri (str): URI for the labeled data.
        tiled_api_key (str, optional): API key for accessing Tiled data.

    Returns:
        tuple:
            - labeled_uris (list[str]): List of labeled URIs.
            - labels (list[str]): List of assigned labels.
    """
    labels_client = from_uri(tiled_uri, api_key=tiled_api_key)
    labels_uids = list(labels_client)

    labeled_uris = []
    labels = []
    for label_uid in labels_uids:
        metadata = labels_client[label_uid].metadata
        labels.append(metadata["label"])
        labeled_uris.append(metadata["uri"])

    return labeled_uris, labels


def normalize_percentiles(x, low_perc=0.01, high_perc=99, mask=None):
    size = tf.shape(x)
    x = tf.reshape(x, [-1])
    if mask is not None:
        mask = tf.reshape(mask, [-1])
    else:
        if x.dtype == tf.uint8:
            zero = tf.constant(0, dtype=tf.uint8)
            mask_nan = tf.math.is_nan(tf.cast(x, tf.float32))
        else:
            zero = tf.constant(0.0, dtype=x.dtype)
            mask_nan = tf.math.is_nan(x)
        # Mask negative and NaN values
        mask_neg = tf.less(x, zero)
        mask = tf.math.logical_or(mask_neg, mask_nan)

    # Apply the mask
    x = tf.boolean_mask(x, tf.logical_not(mask))

    low = tfp.stats.percentile(x, low_perc)
    high = tfp.stats.percentile(x, high_perc)
    x = tf.clip_by_value((x - low) / (high - low), 0, 1) * 255
    x = tf.cast(x, tf.uint8)
    x = tf.reshape(x, size)
    return x


def log_transform(image, threshold=1e-12, low_perc=0.01, high_perc=99, mask=None):
    """
    Apply log transform to the input image
    Args:
        image:          Input image
        threshold:      Threshold value
        low_perc:       Low percentile
        high_perc:      High percentile
        mask:           Mask to be applied
    Returns:
        Image
    """
    # Mask negative and NaN values
    if image.dtype == tf.uint8:
        zero = tf.constant(0, dtype=tf.uint8)
        mask_nan = tf.math.is_nan(tf.cast(image, tf.float32))
    else:
        zero = tf.constant(0.0, dtype=image.dtype)
        mask_nan = tf.math.is_nan(image)
    mask_neg = tf.less(image, zero)

    tmp_mask = tf.math.logical_or(mask_neg, mask_nan)
    if mask is not None:
        mask = tf.math.logical_or(tmp_mask, mask)
    else:
        mask = tmp_mask

    # Apply mask
    image = tf.where(mask, zero, image)

    # Apply log transform
    image = tf.math.log(image + threshold)

    # Apply mask again
    image = tf.where(mask, zero, image)

    # Normalize percentiles
    image = normalize_percentiles(image, low_perc, high_perc, mask)

    return image


def train_tiled_generator(root_uri, tiled_uris, labels, api_key=None):
    """
    Generator function to load Tiled data.

    Args:
        root_uri (str): Root URI from which data should be retrieved.
        tiled_uris (list[str]): List of Tiled URIs (potentially with '?slice=' notation).
        labels (list[Any]): List of labels (parallel to URIs).
        api_key (str, optional): API key for Tiled.

    Yields:
        tuple: (image_tensor, label)
    """
    tiled_client = from_uri(root_uri, api_key=api_key)

    for i, tiled_uri in enumerate(tiled_uris):
        # Ensure tiled_uri starts with root_uri:
        if not tiled_uri.startswith(root_uri):
            raise ValueError(
                f"URI {tiled_uri} does not start with the provided root_uri {root_uri}"
            )

        # Extract the sub_uri from tiled_uri by removing the root_uri prefix
        sub_uri = tiled_uri[len(root_uri) :]

        slice_index = None
        # Check for '?slice=' in the sub_uri. If present, separate the slice index
        if "?slice=" in sub_uri:
            sub_uri, slice_str = sub_uri.split("?slice=", maxsplit=1)
            slice_index = int(slice_str)

        if slice_index is not None:
            block_array = tiled_client[sub_uri][slice_index]
        else:
            block_array = tiled_client[sub_uri].read()

        yield block_array, labels[i]


def inference_tiled_generator(root_uri, sub_uris, api_key=None):
    """
    Generator function to load tiled data
    Args:
        root_uri:       Root URI from which data should be retrieved
        sub_uris:       List of sub URIs
        api_key:        API key for tiled
        labels:         List of labels
    Yields:
        Image tensor
    """
    if isinstance(root_uri, bytes):
        root_uri = root_uri.decode("ascii")
        sub_uris = [sub_uri.decode("ascii") for sub_uri in sub_uris]
    tiled_client = from_uri(root_uri, api_key=api_key)
    for sub_uri in sub_uris:
        block_array = tiled_client[sub_uri]
        if len(block_array.shape) > 2:
            for i in range(block_array.shape[0]):
                yield block_array[i,]
        else:
            yield block_array[:]


def get_dataset(
    data_uris,
    root_uri,
    data_type,
    data_tiled_api_key=None,
    labels_tiled_uri=None,
    labels_tiled_api_key=None,
    shuffle=False,
    seed=42,
):
    """
    Prepare a tf.data.Dataset for training or inference.

    Args:
        data_uris (list[str]): List of URIs for the data (e.g., Tiled URIs or file paths).
        root_uri (str): Root URI for Tiled data retrieval.
        data_type (str): Type of data source ("tiled" or local file type).
        data_tiled_api_key (str, optional): API key for accessing Tiled data.
        labels_tiled_uri (str, optional): URI for labeled data.
        labels_tiled_api_key (str, optional): API key for accessing labeled data.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        seed (int, optional): Random seed for shuffling. Defaults to 42.
    Returns:
        tuple:
            - tf.data.Dataset: The dataset for training or inference.
            - bool: True if the data is a TIFF file (based on file extension), False otherwise.
    """

    # If labels_tiled_uri is provided, load labels
    if labels_tiled_uri:
        labeled_uris, labels = load_labeled_data(labels_tiled_uri, labels_tiled_api_key)

        # Shuffle the data for training and validation split
        if shuffle:
            np.random.seed(seed)
            indices = np.random.permutation(len(labeled_uris))
            labeled_uris = [labeled_uris[i] for i in indices]
            labels = [labels[i] for i in indices]

        # Map labels to integers
        classes = sorted(set(labels))
        class_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
        numeric_labels = [class_map[lbl] for lbl in labels]

        # Create one-hot vectors
        num_classes = len(classes)
        one_hot_labels = tf.keras.utils.to_categorical(numeric_labels, num_classes)

        # Build training dataset (yields (image, label) pairs).

        if data_type == "file":
            # Use the file paths directly
            dataset = tf.data.Dataset.from_tensor_slices((labeled_uris, one_hot_labels))
            file_ext = labeled_uris[0].split(".")[-1].lower()
            is_tif = file_ext in ["tif", "tiff"]
        else:
            is_tif = False
            # Use the Tiled generator function
            # We use partial here so that the generator can receive `None`
            # without TensorFlow trying to convert it into a tensor (which would cause an error).
            dataset = tf.data.Dataset.from_generator(
                partial(
                    train_tiled_generator,
                    root_uri=root_uri,
                    labeled_uris=labeled_uris,
                    one_hot_labels=one_hot_labels,
                    data_tiled_api_key=data_tiled_api_key,  # can be None
                ),
                output_signature=(
                    tf.TensorSpec(shape=(None, None), dtype=tf.float16),
                    tf.TensorSpec(shape=(num_classes,), dtype=tf.float16),
                ),
            )

        return dataset, classes, is_tif, len(labeled_uris)

    # If labels_tiled_uri is not provided, load data only
    else:
        # Build inference dataset (yields images only).
        if data_type == "file":
            # Use the file paths directly
            dataset_files = filepaths_from_directory(root_uri, data_uris)
            dataset = tf.data.Dataset.from_tensor_slices(dataset_files)
            file_ext = dataset_files[0].split(".")[-1].lower()
            is_tif = file_ext in ["tif", "tiff"]
        else:
            is_tif = False
            # Use the Tiled generator function
            # We use partial here so that the generator can receive `None`
            # without TensorFlow trying to convert it into a tensor (which would cause an error).
            dataset = tf.data.Dataset.from_generator(
                inference_tiled_generator,
                partial(
                    inference_tiled_generator,
                    root_uri=root_uri,
                    data_tiled_api_key=data_tiled_api_key,  # can be None
                ),
                output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float16),
            )

        classes = None

        return dataset, classes, is_tif


def tiled_data_preprocessing(
    data,
    target_shape,
    log=False,
    low_perc=0.01,
    high_perc=99,
    mask=None,
):
    """
    Preprocessing function that loads data per batch
    Args:
        data:           Data to be preprocessed
        target_shape:   Target shape of data
        data_type:      Type of data
        log:            Bool indicating if data should be log transformed
        low_perc:       Low percentile
        high_perc:      High percentile
        mask:           Mask to be applied
    Returns:
        image
    """
    # Check number of channels
    if len(data.shape) == 3:
        if data.shape[0] == 3:
            data = tf.transpose(data, perm=[1, 2, 0])
        elif data.shape[0] == 1:
            data = tf.squeeze(data)
        else:
            raise ValueError("Not a valid grayscale image shape")

    if log:
        data = log_transform(data, low_perc=low_perc, high_perc=high_perc, mask=mask)
    else:
        data = normalize_percentiles(
            data, low_perc=low_perc, high_perc=high_perc, mask=mask
        )

    # Convert to RGB
    if len(data.shape) == 3 and data.shape[0] == 1:
        data = tf.transpose(data, (1, 2, 0))
        data = tf.repeat(data, repeats=3, axis=-1)
    elif len(data.shape) == 3 and data.shape[0] == 3:
        data = tf.transpose(data, (1, 2, 0))
    elif len(data.shape) == 2:
        data = tf.expand_dims(data, axis=-1)
        data = tf.repeat(data, repeats=3, axis=-1)
    else:
        raise ValueError("Not a valid image shape")

    data = tf.image.resize(data, tf.constant(target_shape))

    return data


def load_tiff(path):
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    img = iio.imread(path).astype(np.float32)
    return img


def tf_load_tiff(path):
    img = tf.numpy_function(load_tiff, [path], tf.float32)
    img.set_shape([None, None])
    img = tf.expand_dims(img, -1)  # shape: [H, W, 1]
    return img


def file_data_preprocessing(
    img,
    target_shape,
    is_tif,
    log=False,
    low_perc=0.01,
    high_perc=99,
    mask=None,
):
    """
    Preprocessing function that loads data per batch
    Args:
        img:            Image to be preprocessed
        target_shape:   Target shape of data
        is_tif:         Bool indicating if data is in tif format
        log:            Bool indicating if data should be log transformed
        low_perc:       Low percentile
        high_perc:      High percentile
        mask:           Mask to be applied
    Returns:
        image
    """
    img = tf.io.read_file(img)

    if not is_tif:
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
    else:
        img = tf_load_tiff(img)
        # img = tfio.experimental.image.decode_tiff(img)

    if log:
        img = log_transform(img, low_perc=low_perc, high_perc=high_perc, mask=mask)
    else:
        img = normalize_percentiles(
            img, low_perc=low_perc, high_perc=high_perc, mask=mask
        )

    img_3ch = tf.tile(img, [1, 1, 3])  # shape: [H, W, 3]
    img_3ch = tf.image.resize(img_3ch, tf.constant(target_shape))
    img_3ch = tf.cast(img_3ch * 255, tf.uint8)
    return img_3ch


def get_mask(
    mask_tiled_uri=None,
    mask_tiled_api_key=None,
):
    """
    Load mask from Tiled
    Args:
        mask_tiled_uri:     Tiled URI of the mask
        mask_tiled_api_key: API key for Tiled
    Returns:
        Mask as numpy array
    """
    if mask_tiled_uri is None:
        return None
    mask_client = from_uri(mask_tiled_uri, api_key=mask_tiled_api_key)
    return mask_client[:]


def write_results(
    feature_vectors,
    probabilities,
    io_parameters,
    feature_vectors_path,
    probabilities_path,
    metadata=None,
):
    # Prepare Tiled parent node
    uid_save = io_parameters.uid_save
    write_client = from_uri(
        io_parameters.results_tiled_uri, api_key=io_parameters.results_tiled_api_key
    )
    write_client = write_client.create_container(key=uid_save)

    # Save latent vectors to Tiled
    structure = TableStructure.from_pandas(feature_vectors)

    # Remove API keys from metadata
    if metadata:
        metadata["io_parameters"].pop("data_tiled_api_key", None)
        metadata["io_parameters"].pop("results_tiled_api_key", None)
        metadata["io_parameters"].pop("mask_tiled_api_key", None)

    frame = write_client.new(
        structure_family="table",
        data_sources=[
            DataSource(
                structure_family="table",
                structure=structure,
                mimetype="application/x-parquet",
                assets=[
                    Asset(
                        data_uri=f"file://{feature_vectors_path}",
                        is_directory=False,
                        parameter="data_uris",
                        num=1,
                    )
                ],
            )
        ],
        metadata=metadata,
        key="feature_vectors",
    )

    frame.write(feature_vectors)

    # Save probabilities to Tiled
    structure = TableStructure.from_pandas(probabilities)

    frame = write_client.new(
        structure_family="table",
        data_sources=[
            DataSource(
                structure_family="table",
                structure=structure,
                mimetype="application/x-parquet",
                assets=[
                    Asset(
                        data_uri=f"file://{probabilities_path}",
                        is_directory=False,
                        parameter="data_uris",
                        num=1,
                    )
                ],
            )
        ],
        metadata=metadata,
        key="probabilities",
    )

    frame.write(probabilities)

    pass
