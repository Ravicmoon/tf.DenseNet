import tensorflow as tf
import numpy as np
import os
import cv2

def _unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _convert_to_file(dict, file_name, num_samples, is_CIFAR10):
    writer = tf.python_io.TFRecordWriter(file_name)

    for n in range(0, num_samples):

        print('processing ' + str(dict[b'filenames'][n], 'UTF-8') + ' file...')
        
        img = np.reshape(dict[b'data'][n, :], [3, 32, 32])
        img = np.dstack((img[2, :, :], img[1, :, :], img[0, :, :]))
        if is_CIFAR10:
            label = dict[b'labels'][n]
        else:
            label = dict[b'fine_labels'][n]
        '''
        cv2.imshow('CIFAR', img)
        cv2.waitKey()
        '''
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': _bytes_feature(img.tostring()),
            'image/format': _bytes_feature(b'raw'),
            'image/label': _int64_feature(label)}))
    
        writer.write(example.SerializeToString())
    
    writer.close()

'''
 Convert CIFAR-10 dataset to tfrecord files
 Please note that colors are encoded in BGR order, not RGB
'''
for i in range(0, 5):
    
    dict = _unpickle('data_batch_' + str(i + 1))
    _convert_to_file(dict, 'CIFAR-10_train' + str(i) + '.tfrecord', 10000, True)

dict = _unpickle('test_batch')
_convert_to_file(dict, 'CIFAR-10_valid0.tfrecord', 10000, True)

'''
 Convert CIFAR-100 dataset to tfrecord files
'''
dict = _unpickle('train')
_convert_to_file(dict, 'CIFAR-100_train.tfrecord', 50000, False)

dict = _unpickle('test')
_convert_to_file(dict, 'CIFAR-100_valid.tfrecord', 10000, False)