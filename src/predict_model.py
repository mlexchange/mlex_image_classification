import argparse, json, logging, os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from model_validation import DataAugmentationParams
from helper_utils import get_dataset, data_preprocessing
from custom_callbacks import PredictionCustomCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(encoding='utf-8', level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_info', help='path to dataframe of filepaths')
    parser.add_argument('-m', '--model_dir', help='input directory')
    parser.add_argument('-o', '--output_dir', help='output directory')
    parser.add_argument('-p', '--parameters', help='list of training parameters')

    args = parser.parse_args()
    data_parameters = DataAugmentationParams(**json.loads(args.parameters))
    batch_size = data_parameters.batch_size
    logging.info(tf.test.gpu_device_name())

    # Prepare data generators and create a tf.data pipeline of augmented images
    test_dataset, datasets_uris = get_dataset(args.data_info, shuffle=False)
    
    test_generator = test_dataset.map(lambda x: data_preprocessing(x, (224,224)))
    test_generator = test_generator.batch(batch_size)
    with open(args.model_dir+'/class_info.json', 'r') as json_file:
        classes = json.load(json_file)
    class_num = len(classes)

    # Load model and start prediction process
    loaded_model = load_model(args.model_dir+'/model.keras')
    prob = loaded_model.predict(test_generator,
                                verbose=0,
                                callbacks=[PredictionCustomCallback(datasets_uris, classes)])

    df_results = pd.DataFrame(prob, columns=classes)
    df_results.index = datasets_uris

    # Create output directory if it does not exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(f'{args.output_dir}/results.parquet', engine='pyarrow')