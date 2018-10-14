import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

_MEAN_CIFAR10 = [125.3, 123.0, 113.9]
_STD_CIFAR10 = [63.0, 62.1, 66.7]

_MEAN_CIFAR100 = [129.3, 124.1, 112.4]
_STD_CIFAR100 = [68.2, 65.4, 70.4]

class TFRecordDataset:

    def __init__(self, tfrecord_dir, dataset_name):

        self.tfrecord_dir = tfrecord_dir
        self.dataset_name = dataset_name


    def _get_num_samples(self, start_pattern):

        num_samples = 0
        tfrecords_to_count = [os.path.join(self.tfrecord_dir, file) for file in os.listdir(self.tfrecord_dir) 
                              if file.startswith(start_pattern)]
        
        for tfrecord_file in tfrecords_to_count:
            for record in tf.python_io.tf_record_iterator(tfrecord_file):
                num_samples += 1

        return num_samples


    def _get_dataset(self, mode):

        start_pattern = self.dataset_name + '_' + mode

        reader = tf.TFRecordReader;

        keys_to_features, items_to_handlers = self._get_decode_pattern()

        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        return slim.dataset.Dataset(
                data_sources = os.path.join(self.tfrecord_dir, start_pattern + '*'),
                reader = reader,
                decoder = decoder,
                num_samples=self._get_num_samples(start_pattern),
                items_to_descriptions=self._items_to_description())


    def load_batch(self, mode, batch_size=32, height=224, width=224):
        """Loads a single batch of data.

        Args:
          dataset: The dataset to load.
          batch_size: The number of images in the batch.
          height: The size of each image after preprocessing.
          width: The size of each image after preprocessing.

        Returns:
          images: A Tensor of size [batch_size, height, width, 3], preprocessed input images.
          gts: A Tensor of size [batch_size, height, width, 1], annotated images.
        """

        assert(mode in ['train', 'valid'])
    
        dataset = self._get_dataset(mode)
        shuffle = True if mode == 'train' else False
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=shuffle, common_queue_capacity=512, common_queue_min=256)

        return self._preprocess(provider, mode, batch_size, height, width), dataset.num_samples
    

# TFRecord file reader for CIFAR-10 and CIFAR-100 datasets
class TFRecordCIFAR(TFRecordDataset):
    
    def __init__(self, tfrecord_dir, dataset_name, num_labels):

        TFRecordDataset.__init__(self, tfrecord_dir, dataset_name)
        self.num_labels = num_labels


    def _get_decode_pattern(self):

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string),
            'image/format': tf.FixedLenFeature((), tf.string),
            'image/label': tf.FixedLenFeature((), tf.int64),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(shape=[32, 32, 3]),
            'label': slim.tfexample_decoder.Tensor('image/label'),
        }

        return keys_to_features, items_to_handlers


    def _items_to_description(self):

        return {'image': 'An input color image',
                'label': 'A label of the input image'}


    def _normalize_image(self, image, means, stds):
        ''' Revised from vgg_preprocessing.py in TensorFlowOnSpark
        (https://github.com/yahoo/TensorFlowOnSpark/tree/master/examples/slim/preprocessing)
        '''
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        
        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] = (channels[i] - means[i]) / stds[i]
        return tf.concat(axis=2, values=channels)


    def _preprocess(self, provider, mode, batch_size, height, width):

        [image, label] = provider.get(['image', 'label'])
        
        image = tf.cast(image, tf.float32)
        if self.dataset_name == 'CIFAR-10':
            image = self._normalize_image(image, _MEAN_CIFAR10, _STD_CIFAR10)
        elif self.dataset_name == 'CIFAR-100':
            image = self._normalize_image(image, _MEAN_CIFAR100, _STD_CIFAR100)
        else:
            image = tf.image.per_image_standardization(image)

        if mode == 'train':
            paddings = tf.constant([[4, 4], [4, 4], [0, 0]])
            image = tf.pad(image, paddings, 'CONSTANT')
            image = tf.random_crop(image, [32, 32, 3])
            image = tf.image.random_flip_left_right(image)

        one_hot_label = slim.one_hot_encoding(label, self.num_labels)

        images, labels = tf.train.batch([image, one_hot_label], batch_size=batch_size, capacity=2*batch_size)

        return images, labels