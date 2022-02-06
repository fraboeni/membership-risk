import os
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import minmax_scale
from code_base.get_data import get_data, data_augmentation
from code_base.models import *
from keras.regularizers import l2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
AUTOTUNE = tf.data.AUTOTUNE


def main(dataset='cifar10', model_name='cifar10_resnet50', augment=False, batch_size=64, lr=0.001, optim="Adam",
         momentum=0.9, nesterov=False, epochs=50, early_stop=True, save_model=True, log_training=True,
         logdir='log_dir/models/', from_logits=True, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001),
         data_dir='log_dir/result_data/experiments_64-20220204T144927Z-001/experiments_64'):
    # when data dir is not specified, we use the keras dataset, otherwise, we load data from disk
    if data_dir is None:
        train_data, test_data = get_data(dataset, augmentation=augment, batch_size=batch_size,
                                         indices_to_use=range(0, 25000))
    else:
        load_location = os.getcwd() + '/../' + data_dir
        x = np.load(load_location + "/x_train.npy")
        x_scaled = minmax_scale(x.flatten(), feature_range=(0, 1)).reshape(50000, 32, 32, 3)  # scale to range(0,1)
        y = np.load(load_location + "/y_train.npy")

        # todo: always chose different samples
        # keep = np.random.uniform(0, 1, size=())
        train_data = tf.data.Dataset.from_tensor_slices((x_scaled[:25000], y[:25000]))
        test_data = tf.data.Dataset.from_tensor_slices((x_scaled[25000:], y[25000:]))

        train_data = train_data.shuffle(25000).batch(batch_size)
        test_data = test_data.batch(batch_size)

        if augment:
            train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y),
                                        num_parallel_calls=AUTOTUNE)

    if optim == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr,
                                            momentum=momentum,
                                            nesterov=nesterov)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if from_logits:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model = MODELS[model_name](from_logits=from_logits)  # todo: add ability to pass base trainable to resnet model

    if kernel_regularizer is not None:
        model = add_weight_decay(model, kernel_regularizer, bias_regularizer)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy']
                  )
    model_id = 1013  # Todo parse from config file

    callbacks = []

    if early_stop:
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        callbacks.append(early_stop_callback)

    if log_training:
        logfile = os.getcwd() + '/../' + logdir + dataset + '/' + str(model_id) + '_' + model_name + '.csv'
        print(logfile)
        logging_callback = tf.keras.callbacks.CSVLogger(logfile, separator=",", append=False)
        callbacks.append(logging_callback)

    history = model.fit(train_data,
                        validation_data=test_data,
                        epochs=epochs,
                        callbacks=callbacks,
                        )
    print(history.history)

    if save_model:
        model.save(os.getcwd() + '/../' + logdir + dataset + '/' + str(model_id) + '_' + model_name, save_format='tf')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', choices=['cifar10', 'fmnist'])
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam"])
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--early_stop', type=bool, default=True)
    # parser.add_argument('--logdir', default=None)
    # args = parser.parse_args()
    main()  # **vars(args))
