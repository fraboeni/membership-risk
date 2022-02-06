import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from sklearn.preprocessing import minmax_scale

AUTOTUNE = tf.data.AUTOTUNE
SHAPES = {
    "cifar10": (32, 32, 3),
    "fmnist": (28, 28, 1),
}


def get_data(name, augmentation=False, batch_size=64, indices_to_use=None, data_dir=None):
    """
    Helper function to load the datasets.
    :param data_dir: str
        location where the data should be loaded from disk
    :param name: str
        name of dataset to be loaded
    :param augmentation: bool
        whether to perform data augmentation
    :param indices_to_use: list of int
        if specified tells which data points of train data to include, if datadir is None, then test data is separate CIFAR10
        test data, if a datadir is given, then the test data corresponds to the non_members
    :return: ndarray
        train and test data and labels in arrays
    """

    if name == "cifar10":

        if data_dir is None:
            train, test = tf.keras.datasets.cifar10.load_data()
            train, train_labels = train
            test, test_labels = test

            # include only the data points from the specific indices
            if indices_to_use is not None:
                train, train_labels = train[indices_to_use], train_labels[indices_to_use]
                print(train.shape)

            train_labels = train_labels.flatten()
            test_labels = test_labels.flatten()

            if augmentation:
                # augmentation from: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
                # data generator from: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
                train_datagen = ImageDataGenerator(rescale=1. / 255,
                                                   width_shift_range=0.1,
                                                   height_shift_range=0.1,
                                                   horizontal_flip=True,
                                                   rotation_range=10,
                                                   )
                train = train_datagen.flow(
                    train, train_labels,
                    batch_size=batch_size)

                test_datagen = ImageDataGenerator(rescale=1. / 255,
                                                  )
                test = test_datagen.flow(
                    test, test_labels,
                    batch_size=batch_size)
            else:
                train_datagen = ImageDataGenerator(rescale=1. / 255)
                train = train_datagen.flow(
                    train, train_labels,
                    batch_size=batch_size)

                test_datagen = ImageDataGenerator(rescale=1. / 255)
                test = test_datagen.flow(
                    test, test_labels,
                    batch_size=batch_size)

        else: # if we should load data from disk
            load_location = os.getcwd() + '/../' + data_dir
            x = np.load(load_location + "/x_train.npy")
            x_scaled = minmax_scale(x.flatten(), feature_range=(0, 1)).reshape(50000, 32, 32, 3)  # scale to range(0,1)
            y = np.load(load_location + "/y_train.npy")

            train = tf.data.Dataset.from_tensor_slices((x_scaled[:25000], y[:25000]))
            test = tf.data.Dataset.from_tensor_slices((x_scaled[25000:], y[25000:]))

            train = train.shuffle(25000).batch(batch_size)
            test = test.batch(batch_size)

            if augmentation:
                train = train.map(lambda x, y: (data_augmentation(x, training=True), y),
                                            num_parallel_calls=AUTOTUNE)

    else:
        raise ValueError(f"unknown dataset {name}")

    return train, test


def data_augmentation():
    data_augmentation_net = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1)

    ])
    return data_augmentation_net
