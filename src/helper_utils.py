import os

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SPLASH_CLIENT = 'http://splash:80/api/v0'


def load_from_splash(uri_list, event_id):
    '''
    This function queries labels from splash-ml.
    Args:
        uri_list:    URI of dataset (e.g. file path)
        event_id:    Tagging event id in splash_ml
    Returns:
        splash_df:   Dataframe of labeled images (docker path)
    '''
    url = f'{SPLASH_CLIENT}/datasets/search'
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


def get_dataset(data, shuffle=False, event_id = None, seed=42):
    '''
    This function prepares the dataset to be used during training and/or prediction processes
    Input:
        data:           Path to parquet file with the list of data
        shuffle:        Bool indicating if the dataset should be shuffled
        event_id:       Tagging event id in splash_ml
        seed:           Seed for random number generation
    Returns:
        TF dataset, classes or filenames
    '''
    # Retrieve data set list
    data_info = pd.read_parquet(data, engine='pyarrow')
    if 'local_uri' in data_info:
        uri_list = data_info['local_uri']
    else:
        uri_list = data_info['uri']
    # Retrieve labels
    if event_id:
        labeled_uris, labels = load_from_splash(uri_list.tolist(), event_id)
        classes = list(set(labels))
        df_labels = pd.DataFrame(labels).replace({class_name: label for label, class_name in 
                                                  enumerate(classes)})
        categorical_labels = tf.keras.utils.to_categorical(df_labels.iloc[:, 0].tolist(),
                                                           num_classes = len(classes))
        num_imgs = len(labeled_uris)
        dataset = tf.data.Dataset.from_tensor_slices((labeled_uris, categorical_labels))
        kwarg = classes
    else:
        kwarg = uri_list.to_list()
        dataset = tf.data.Dataset.from_tensor_slices(uri_list)
        num_imgs = len(uri_list)
    # Shuffle data
    if shuffle:
        dataset.shuffle(seed=seed, buffer_size=num_imgs)
    return dataset, kwarg


def data_preprocessing(file_path, target_shape):
    '''
    Preprocessing function that loads data per batch
    Args:
        file_path:      Path to file
        target_shape:   Target shape of data
    Returns:
        image
    '''
    img = tf.io.read_file(file_path)
    try:
        img = tf.io.decode_image(img, channels=3, expand_animations = False)
    except:
        img = tf.io.experimental.image.decode_tiff(img, expand_animations = False)
        img = tf.io.experimental.color.rgba_to_rgb(img)
    img = tf.image.resize(img, tf.constant(target_shape))
    img = img / 255.
    return img