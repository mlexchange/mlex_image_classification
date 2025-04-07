from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf

from src.helper_utils import (
    file_data_preprocessing,
    filepaths_from_directory,
    get_dataset,
    inference_tiled_generator,
    load_labeled_data,
    log_transform,
    normalize_percentiles,
    tiled_data_preprocessing,
    train_tiled_generator,
)


def glob_side_effect(path_pattern, recursive=False):
    if "cache" in path_pattern:
        if path_pattern.endswith("*.[pP][nN][gG]"):
            return ["cache/file1.png"]
        return []
    if path_pattern.endswith("*.[pP][nN][gG]"):
        return ["file1.png"]
    elif path_pattern.endswith("*.[jJ][pP][gG]") or path_pattern.endswith(
        "*.[jJ][pP][eE][gG]"
    ):
        return ["file2.jpg"]
    return []


@patch("glob.glob", side_effect=glob_side_effect)
@patch("os.path.isdir")
def test_filepaths_from_directory(mock_isdir, mock_glob):
    root_uri = "/test"
    selected_sub_uris = [""]
    result = filepaths_from_directory(root_uri, selected_sub_uris)
    assert result == ["file1.png", "file2.jpg"]


@patch("src.helper_utils.from_uri")
def test_load_labeled_data(mock_from_uri):
    mock_client = MagicMock()
    mock_client.__iter__.return_value = ["uid1", "uid2"]
    mock_client.__getitem__.side_effect = [
        MagicMock(metadata={"label": "cat", "uri": "uri1"}),
        MagicMock(metadata={"label": "dog", "uri": "uri2"}),
    ]
    mock_from_uri.return_value = mock_client
    tiled_uri = "http://example.com"
    labeled_uris, labels = load_labeled_data(tiled_uri)
    assert labeled_uris == ["uri1", "uri2"]
    assert labels == ["cat", "dog"]


def test_normalize_percentiles():
    x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
    result = normalize_percentiles(x)
    expected = tf.constant([0, 63, 127, 191, 255], dtype=tf.uint8)
    tf.debugging.assert_equal(result, expected)


def test_log_transform():
    image = tf.constant([1.0, 10.0, 100.0], dtype=tf.float32)
    result = log_transform(image, threshold=1e-12, low_perc=0, high_perc=100)
    assert result.dtype == tf.uint8


@patch("src.helper_utils.from_uri")
def test_train_tiled_generator(mock_from_uri):
    mock_client = MagicMock()
    mock_client.__getitem__.return_value.read.return_value = np.array([1, 2, 3])
    mock_from_uri.return_value = mock_client
    root_uri = "http://example.com"
    tiled_uris = ["http://example.com/uri1"]
    labels = ["label1"]
    generator = train_tiled_generator(root_uri, tiled_uris, labels)
    image, label = next(generator)
    np.testing.assert_array_equal(image, np.array([1, 2, 3]))
    assert label == "label1"


@patch("src.helper_utils.from_uri")
def test_inference_tiled_generator(mock_from_uri):
    mock_client = MagicMock()
    mock_client.__getitem__.return_value = np.array([1, 2, 3])
    mock_from_uri.return_value = mock_client
    root_uri = "http://example.com"
    sub_uris = ["uri1"]
    generator = inference_tiled_generator(root_uri, sub_uris)
    result = next(generator)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


@patch("src.helper_utils.filepaths_from_directory")
def test_get_dataset(mock_filepaths_from_directory):
    mock_filepaths_from_directory.return_value = ["file1.png", "file2.png"]
    data_uris = ["subdir"]
    root_uri = "/test"
    data_type = "file"
    dataset, classes, is_tif = get_dataset(data_uris, root_uri, data_type)
    assert isinstance(dataset, tf.data.Dataset)
    assert classes is None
    assert not is_tif


def test_tiled_data_preprocessing():
    data = tf.random.uniform((1, 100, 100), dtype=tf.float32)
    target_shape = (64, 64)
    result = tiled_data_preprocessing(data, target_shape, log=True)
    assert result.shape == (64, 64, 3)


def test_file_data_preprocessing():
    img = tf.random.uniform((100, 100, 1), dtype=tf.float32)
    target_shape = (64, 64)
    is_tif = False
    with (
        patch("tensorflow.io.read_file", return_value=img),
        patch("tensorflow.io.decode_image", return_value=img),
    ):
        result = file_data_preprocessing("dummy_path", target_shape, is_tif, log=True)
        assert result.shape == (64, 64, 3)
