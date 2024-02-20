import argparse, json, logging, os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from model_validation import DataAugmentationParams, model_list
from helper_utils import get_dataset, data_preprocessing
from custom_callbacks import PredictionCustomCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(encoding='utf-8', level=logging.INFO)


# Prepare the trained model to return both the probabilities and feature vectors
class CustomModel(tf.keras.Model):
    def __init__(self, trained_model):
        super(CustomModel, self).__init__()
        self.base_model = tf.keras.Model(
            inputs = trained_model.input,
            outputs = trained_model.layers[-2].output)
        self.last_layers = tf.keras.Model(
            inputs = trained_model.layers[-2].output,
            outputs = trained_model.output)
    
    def call(self, inputs, training=False):
        second_to_last = self.base_model(inputs)
        x = self.last_layers(second_to_last)
        return x, second_to_last


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_info', help='path to dataframe of filepaths')
    parser.add_argument('-m', '--model_dir', help='input directory')
    parser.add_argument('-o', '--output_dir', help='output directory')
    parser.add_argument('-p', '--parameters', help='list of prediction parameters')

    args = parser.parse_args()
    data_parameters = DataAugmentationParams(**json.loads(args.parameters))
    batch_size = data_parameters.batch_size
    logging.info(tf.test.gpu_device_name())

    # Load trained model and parameters
    loaded_model = load_model(args.model_dir+'/model.keras')
    model_name, *weights = loaded_model._name.split('_')
    target_size = model_list[model_name]
    custom_model = CustomModel(loaded_model)    # Modify trained model to return prob and f_vec

    # Prepare data generators and create a tf.data pipeline of augmented images
    predict_dataset, datasets_uris, data_type = get_dataset(args.data_info, shuffle=False)
    predict_generator = predict_dataset.map(lambda x: data_preprocessing(x, (target_size,target_size), data_type, data_parameters.log))

    # Preprocess input according to the model if weights are set to imagenet
    if weights == ['imagenet']:
        preprocess_input = getattr(tf.keras.applications, model_name).preprocess_input
        predict_generator = predict_dataset.batch(batch_size).map(lambda x: (preprocess_input(x)))
    else:
        predict_generator = predict_generator.batch(batch_size)
    
    with open(args.model_dir+'/class_info.json', 'r') as json_file:
        classes = json.load(json_file)
    class_num = len(classes)

    # Start prediction process    
    prob, f_vec = custom_model.predict(predict_generator,
                                       verbose=0,
                                       callbacks=[PredictionCustomCallback(datasets_uris, classes)])

    df_results = pd.DataFrame(prob, columns=classes)
    df_results.index = datasets_uris

    df_f_vec = pd.DataFrame(f_vec)
    df_f_vec.columns = df_f_vec.columns.astype(str)
    df_f_vec.index = datasets_uris

    # Create output directory if it does not exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(f'{args.output_dir}/results.parquet', engine='pyarrow')
    df_f_vec.to_parquet(f'{args.output_dir}/f_vectors.parquet', engine='pyarrow')