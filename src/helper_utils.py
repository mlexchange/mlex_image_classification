import os
import io

import numpy as np
import pandas as pd
from PIL import Image
import requests

import tensorflow as tf
import tensorflow_io as tfio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SPLASH_URL = 'http://splash:80/api/v0'


def load_from_splash(uri_list, event_id):
    '''
    This function loads existing labels from splash-ml
    Args:
        uri_list:       URI of dataset (e.g. file path)
        event_id:       Tagging event id in splash_ml
    Returns:
        labeled_uris:   List of labeled URIs
        labels:         List of assigned labels
    '''
    url = f'{SPLASH_URL}/datasets/search'
    params = {"page[offset]": 0, "page[limit]": len(uri_list)}
    data = {'uris': uri_list, 'event_id': event_id}
    datasets = requests.post(url, params=params, json=data).json()
    labeled_uris = []
    labels = []
    for dataset in datasets:
        for tag in dataset['tags']:
            if tag['event_id'] == event_id:
                labels.append(tag['name'])
                labeled_uris.append(dataset['uri'])
    return labeled_uris, labels


def parse_tiled(uri, log=False):
    '''
    Parse function to load tiled data
    Args:
        uri:                URI from which data should be retrieved
        log:                Bool indicating if data should be log transformed
    Returns:
        Image
    '''
    uri = uri.decode('utf-8')
    tiled_uri, metadata = uri.split('&expected_shape=')
    # Check if the data is in the expected shape
    expected_shape= metadata.split('&dtype=')[0]
    expected_shape = np.array(list(map(int, expected_shape.split('%2C'))))
    if len(expected_shape) == 3 and expected_shape[0] in [1,3,4]:
        expected_shape = expected_shape[[1,2,0]]
    elif len(expected_shape) != 2 or expected_shape[-1] not in [1,3,4]:
        raise RuntimeError(f"Not supported type of data. Tiled uri: {tiled_uri} and data dimension {expected_shape}")
    # Get data from tiled URI
    contents = get_tiled_response(tiled_uri, expected_shape, max_tries=5)
    image = Image.open(io.BytesIO(contents)).convert("L")
    if log:
        image = np.log1p(np.array(image))
        image = (((image - np.min(image)) / (np.max(image) - np.min(image)))* 255).astype(np.uint8)
        image = Image.fromarray(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def get_tiled_response(tiled_uri, expected_shape, max_tries=5):
    '''
    Get response from tiled URI
    Args:
        tiled_uri:          Tiled URI from which data should be retrieved
        expected_shape:     Expected shape of data
        max_tries:          Maximum number of tries to retrieve data, defaults to 5
    Returns:
        Response content
    '''
    status_code = 502
    trials = 0
    while status_code != 200 and trials < max_tries:
        if len(expected_shape) == 3:
            response = requests.get(f'{tiled_uri},0,:,:&format=png')
        else:
            response = requests.get(f'{tiled_uri},:,:&format=png')
        status_code = response.status_code
        trials += 1
    if status_code != 200:
        raise Exception(f'Failed to retrieve data from {tiled_uri}')
    return response.content


def gen(uri_list, log=False):
    '''
    Generator function to load tiled data
    Args:
        uri_list:           List of URIs from which data should be retrieved
        log:                Bool indicating if data should be log transformed
    Returns:
        Image tensor
    '''
    for uri in uri_list:
        image = parse_tiled(uri, log)
        image_tensor = tf.convert_to_tensor(np.array(image))
        yield image_tensor


def get_dataset(data, shuffle=False, event_id = None, seed=42):
    '''
    This function prepares the dataset to be used during training and/or prediction processes
    Input:
        data:           Path to parquet file with the list of data
        shuffle:        Bool indicating if the dataset should be shuffled
        event_id:       Tagging event id in splash_ml
        seed:           Seed for random number generation
    Returns:
        TF dataset, kwargs (classes or filenames), data_type
    '''
    # Retrieve data set list
    data_info = pd.read_parquet(data, engine='pyarrow')
    # Retrieve labels
    if event_id:
        # Training
        if 'local_uri' in data_info:
            uri_list = data_info['local_uri']
            splash_uri_list = data_info['uri']
            splash_labeled_uris, labels = load_from_splash(splash_uri_list.tolist(), event_id)
            labeled_uris = data_info[data_info['uri'].isin(splash_labeled_uris)]
            labeled_uris = list(labeled_uris['local_uri'])
        else:
            uri_list = data_info['uri']
            labeled_uris, labels = load_from_splash(uri_list.tolist(), event_id)
        classes = list(set(labels))
        df_labels = pd.DataFrame(labels).replace({class_name: label for label, class_name in 
                                                  enumerate(classes)})
        categorical_labels = tf.keras.utils.to_categorical(df_labels.iloc[:, 0].tolist(),
                                                           num_classes = len(classes))
        num_imgs = len(labeled_uris)
        dataset = tf.data.Dataset.from_tensor_slices((labeled_uris, categorical_labels))
        kwargs = classes
    else:
        # Inference
        uri_list = data_info['uri']
        kwargs = uri_list.to_list()
        if data_info['type'][0] == 'tiled':
            dataset = tf.data.Dataset.from_generator(
                gen,
                args=(uri_list,),
                output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float32)
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(uri_list)
            num_imgs = len(uri_list)
    # Check if data is in tif format or tiled
    if data_info['type'][0] == 'tiled' and not event_id:
        data_type = 'tiled'
    elif uri_list[0].split('.')[-1] in ['tif', 'tiff', 'TIF', 'TIFF'] or data_info['type'][0] == 'tiled':
        data_type = 'tif'
    else:
        data_type = 'non-tif'
    # Shuffle data
    if shuffle:
        dataset = dataset.shuffle(seed=seed, buffer_size=num_imgs)
    return dataset, kwargs, data_type


def data_preprocessing(data, target_shape, data_type, log=False, threshold=1e-6):
    '''
    Preprocessing function that loads data per batch
    Args:
        data:           Data to be preprocessed
        target_shape:   Target shape of data
        data_type:      Type of data
        log:            Bool indicating if data should be log transformed
    Returns:
        image
    '''
    if data_type != 'tiled':
        img = tf.io.read_file(data)
        if data_type != 'tif':
            img = tf.io.decode_image(img, channels=3, expand_animations = False)
        else:
            img_tmp = tfio.experimental.image.decode_tiff(img)
            r, g, b = img_tmp[:, :, 0], img_tmp[:, :, 1], img_tmp[:, :, 2]
            img = tf.stack([r, g, b], axis=-1)
    else:
        img = tf.expand_dims(data, axis=-1)
        img = tf.repeat(img, repeats=3, axis=-1)
    img = tf.image.resize(img, tf.constant(target_shape))
    img = img / 255.
    if log:
        img = tf.math.log(img+threshold)
        img = (((img - tf.math.reduce_min(img)) / (tf.math.reduce_max(img) - tf.math.reduce_min(img))) * 255)
        img = tf.cast(img, tf.uint8)
    return img