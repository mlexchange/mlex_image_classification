import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

class TrainCustomCallback(tf.keras.callbacks.Callback):
    '''
    Keras callback for model training. Threads while keras functions are running
    so that you can see training progress
    '''
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss'):
            if epoch == 0:
                logger.info('epoch,loss,val_loss,accuracy,val_accuracy')
            loss = logs.get('loss')
            val_loss = logs.get('val_loss')
            accuracy = logs.get('accuracy')
            val_accuracy = logs.get('val_accuracy')
            logger.info(f'{epoch},{loss},{val_loss},{accuracy},{val_accuracy}')
        else:
            if epoch == 0:
                logger.info('epoch,loss,accuracy')
            loss = logs.get('loss')
            accuracy = logs.get('accuracy')
            logger.info(f'{epoch},{loss},{accuracy}')

    def on_train_end(self, logs=None):
        logger.info('Train process completed')


class PredictionCustomCallback(tf.keras.callbacks.Callback):
    '''
    Keras callback for model prediction. Threads while keras functions are running
    so that you can see evaluation progress
    '''
    def __init__(self, filenames=None, classes=None):
        self.classes = classes
        self.filenames = filenames

    def on_predict_begin(self, logs=None):
        logger.info('Prediction process started')

    def on_predict_batch_end(self, batch, logs=None):
        out = logs['outputs'][0]
        batch_size = out.shape[0]
        if batch==0:
            msg = ['filename'] + self.classes
            logger.info(','.join(msg))
        filenames = self.filenames[batch*batch_size:(batch+1)*batch_size]
        for row in range(batch_size):         # when batch>1
            prob = np.char.mod('%f', out[row,:])
            logger.info(f"{filenames[row]},{','.join(prob)}")

    def on_predict_end(self, logs=None):
        logger.info('Prediction process completed')