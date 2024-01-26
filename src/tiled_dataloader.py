import tensorflow as tf
import requests


class CustomTiledDataset(tf.data.Dataset):
    def __init__(self, uri_list, log):
        self.uri_list = uri_list
        self.log = log
        self.dataset = tf.data.Dataset.from_tensor_slices(self.uri_list)

    @staticmethod
    def _get_tiled_response(tiled_uri, expected_shape, max_tries=5):
        '''
        Get response from tiled URI
        Args:
            tiled_uri:          Tiled URI from which data should be retrieved
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

    def _parse_function(self, uri):
        tiled_uri, metadata = tf.strings.split(uri, '&expected_shape=')
        expected_shape = tf.strings.split(metadata, '&dtype=')[0]
        expected_shape = tf.strings.split(expected_shape, '%2C')
        expected_shape = tf.strings.to_number(expected_shape, out_type=tf.int32)
        expected_shape = tf.cond(tf.equal(tf.shape(expected_shape)[0], 3) & tf.reduce_any(tf.equal(expected_shape[0], [1,3,4])),
                                 lambda: expected_shape[[1,2,0]],
                                 lambda: expected_shape)
        contents = self._get_tiled_response(tiled_uri, expected_shape, max_tries=5)
        image = tf.io.decode_image(contents, channels=1)
        if self.log:
            image = tf.math.log1p(tf.cast(image, tf.float32))
            image = ((image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))) * 255
            image = tf.cast(image, tf.uint8)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def __new__(cls, uri_list, log):
        instance = super(CustomTiledDataset, cls).__new__(cls)
        instance.__init__(uri_list, log)
        return tf.data.Dataset.map(instance.dataset, instance._parse_function)