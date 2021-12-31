import os
import pickle
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



SHAPES = {
    "cifar10": (32, 32, 3),
    "fmnist": (28, 28, 1),
}

def get_data(name, augmentation=False, batch_size=64, indices_to_use=None):
    """
    Helper function to load the datasets.
    :param name: str
        name of dataset to be loaded
    :param augmentation: bool

    :param indices_to_use: list of int
        if specified tells which data points of train data to include, test data is always completely returned
    :return: ndarray
        train and test data and labels in arrays
    """

    if name == "cifar10":

        train, test = tf.keras.datasets.cifar10.load_data()
        train_data, train_labels = train
        test_data, test_labels = test

        # include only the data points from the specific indices
        if indices_to_use is not None:
            train_data, train_labels = train_data[indices_to_use], train_labels[indices_to_use]
            print(train_data.shape)

        train_labels = train_labels.flatten()
        test_labels = test_labels.flatten()

        if augmentation:
            # augmentation from: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
            # data generator from: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               width_shift_range=0.1,
                                               height_shift_range=0.1,
                                               horizontal_flip=True,
                                               )
            train_generator = train_datagen.flow(
                train_data, train_labels,
                batch_size=batch_size)

            test_datagen = ImageDataGenerator(rescale=1. / 255,
                                              width_shift_range=0.1,
                                              height_shift_range=0.1,
                                              horizontal_flip=True,
                                              )
            test_generator = test_datagen.flow(
                test_data, test_labels,
                batch_size=batch_size)
        else:
            train_datagen = ImageDataGenerator(rescale=1. / 255)
            train_generator = train_datagen.flow(
                train_data, train_labels,
                batch_size=batch_size)

            test_datagen = ImageDataGenerator(rescale=1. / 255)
            test_generator = test_datagen.flow(
                test_data, test_labels,
                batch_size=batch_size)



    # elif name == "fmnist":
    #
    #     train, test = tf.keras.datasets.fashion_mnist.load_data()
    #     train_data, train_labels = train
    #     test_data, test_labels = test
    #
    #     train_data = np.array(train_data, dtype=np.float32) / 255
    #     test_data = np.array(test_data, dtype=np.float32) / 255
    #
    #     train_data = train_data.reshape(train_data.shape[0], SHAPES["fmnist"][0], SHAPES["fmnist"][1], SHAPES["fmnist"][2])
    #     test_data = test_data.reshape(test_data.shape[0], SHAPES["fmnist"][0], SHAPES["fmnist"][1], SHAPES["fmnist"][2])

    else:
        raise ValueError(f"unknown dataset {name}")

    return train_generator, test_generator