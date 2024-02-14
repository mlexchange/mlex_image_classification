import argparse, json, logging, os

import tensorflow as tf
import tensorflow.keras.layers as layers

from model_validation import TrainingParams, DataAugmentationParams, model_list
from helper_utils import get_dataset, data_preprocessing
from custom_callbacks import TrainCustomCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(encoding='utf-8', level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_info', help='path to dataframe of filepaths')
    parser.add_argument('-e', '--event_id', help='tagging event id in splash-ml')
    parser.add_argument('-o', '--output_dir', help='output directory')
    parser.add_argument('-p', '--parameters', help='list of training parameters')

    args = parser.parse_args()
    train_parameters = TrainingParams(**json.loads(args.parameters))
    data_parameters = DataAugmentationParams(**json.loads(args.parameters))
    logging.info(tf.test.gpu_device_name())

    # Gather data preprocessing parameters
    batch_size = data_parameters.batch_size
    image_flip = data_parameters.image_flip
    rotation_angle = data_parameters.rotation_angle
    val_pct = data_parameters.val_pct
    seed = data_parameters.seed

    if image_flip.value != 'None' and rotation_angle is not None:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip(image_flip.value, seed=seed),
            layers.RandomRotation(rotation_angle, seed=seed),
            ])
    elif image_flip.value != 'None':
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip(image_flip.value, seed=seed)
            ])
    elif rotation_angle is not None:
        data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(rotation_angle, seed=seed)
        ])
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
    dataset, classes, data_type = get_dataset(args.data_info, 
                                              seed=seed, 
                                              shuffle=data_parameters.shuffle, 
                                              event_id=args.event_id)

    val_size = int(len(dataset)*val_pct/100)
    train_size = len(dataset) - val_size
    logging.info(f"Train size: {train_size}, Validation size: {val_size}")

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    target_size = model_list[train_parameters.nn_model.name]

    train_generator = train_dataset.map(lambda x, y: (data_preprocessing(x, (target_size,target_size), data_type, data_parameters.log), y))
    val_generator = val_dataset.map(lambda x, y: (data_preprocessing(x, (target_size,target_size), data_type, data_parameters.log), y))

    train_generator = train_generator.batch(batch_size).map(lambda x, y: (data_augmentation(x), y))
    val_generator = val_generator.batch(batch_size).map(lambda x, y: (data_augmentation(x), y))
    class_num = len(classes)

    # Define optimizer
    opt_code = compile(f'tf.keras.optimizers.{optimizer}(learning_rate={learning_rate})', 
                       '<string>', 'eval')

    logging.info(f'weights: {weights}')
    if weights != 'None':
        # Load pretrained weights
        model_description = f"tf.keras.applications.{nn_model}(include_top=False, \
            input_shape=({target_size},{target_size},3), weights='imagenet', input_tensor=None)"
        model_code = compile(model_description, "<string>", 'eval')
        base_model = eval(model_code)
        # Adapt output of the model according to the number of classes
        x = base_model.output
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(4096, activation="relu", name="fc1")(x)
        x = layers.Dense(4096, activation="relu", name="fc2")(x)
        predictions = layers.Dense(class_num, activation='softmax', name="predictions")(x)
        model = tf.keras.models.Model(inputs=base_model.input,
                                      outputs=predictions,
                                      name=base_model._name)
    else:
        model_description = f"tf.keras.applications.{nn_model}(include_top=True, weights=None, \
                             input_tensor=None, classes={class_num})"
        model_code = compile(model_description, "<string>", 'eval')
        model = eval(model_code)

    # Compile ML model
    model.compile(optimizer=eval(opt_code),
                  loss=loss_func,
                  metrics=['accuracy'])
    # Print model summary
    model.summary()

    # fit model while also keeping track of data for dash plots
    model.fit(train_generator,
              validation_data=val_generator,
              epochs=epochs,
              verbose=0,
              callbacks=[TrainCustomCallback()],
              shuffle=data_parameters.shuffle)

    # save model
    model.save(args.output_dir+'/model.keras')
    with open(args.output_dir+'/class_info.json', 'w') as json_file:
        json.dump(classes, json_file)
    logging.info("Training process completed")