import os
import tensorflow as tf

from code_base.get_data import get_data
from code_base.models import MODELS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(dataset='cifar10', model_name='cifar10_cnn', augment=True, batch_size=64, lr=0.001, optim="Adam",
         momentum=0.9, nesterov=False, epochs=50, early_stop=True, save_model=True, log_training=True,
         logdir='log_dir/models/', from_logits=True):

    train_data, test_data = get_data(dataset, augmentation=augment, batch_size=batch_size, indices_to_use=range(0,25000))

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
    model = MODELS[model_name](from_logits=from_logits)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy']
                  )
    model_id = 1004 #Todo parse from config file

    callbacks = []

    if early_stop:
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
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
