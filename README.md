# tf.DenseNet
Tensorflow implementation of [DenseNet](http://arxiv.org/abs/1608.06993)

There are three objectives of this project
1. Mimic original Torch implementation
2. Implement similar interfaces of networks in __tf.contrib.slim.nets__
3. Use hihg-level API of Tensorflow

## Prerequisites
1. Python 3.6
2. Tensorflow rc1.10 or above
3. Numpy
4. CIFAR-10 or CIFAR-100 datasets

## Preparing dataset
Please download CIFAR-10 or CIFAR-100 datasets from [here](https://www.cs.toronto.edu/~kriz/cifar.html). Use _CIFAR_to_tfrecord.py_ to convert CIFAR datasets to _tfrecord_ files.

## Training and evaluation
Use _CIFAR_trainval.py_ to train or evalulate the DenseNet architecture on CIFAR datasets.

## Results on CIFAR-10
Summary of the experiment
* DenseNet-BC with _L_=100 and _k_=12
* Data augmentation (shifting and horizontal mirroring)
* ~300 epochs (240,000 iterations) and batch size of 64
* 0.9 Nesterov momentum
* Learning rates: until 50%&rarr;0.1, until 75%&rarr;0.01, after 75%&rarr;0.001

Achieved an __error rate of 5.74%__

## Important notes
Currently, we have tested the implementation only for CIFAR-10. More results will be posted right after we finish the experiments.
