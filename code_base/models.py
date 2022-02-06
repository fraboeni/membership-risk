import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model


def cifar10_cnn(from_logits=True):
    shape = (32, 32, 3)
    i = Input(shape=shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3))(
        i
    )
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    if from_logits:
        x = Dense(10)(x)
    else:
        x = Dense(10, activation="softmax")(x)
    model = Model(i, x)
    return model


def cifar10_resnet50(from_logits=True, base_trainable=False):
    feature_extractor = tf.keras.applications.resnet.ResNet50(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    shape = (32, 32, 3)
    i = Input(shape=shape)
    x = UpSampling2D(size=(7, 7))(
        i
    )  # upsample 32, 32 to 224, 224 by multiplying with factor 7
    if not base_trainable:
        feature_extractor.trainable = False
    x = feature_extractor(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    if from_logits:
        x = Dense(10)(x)
    else:
        x = Dense(10, activation="softmax")(x)
    model = Model(i, x)
    return model


def add_weight_decay(model, kernel_regularizer, bias_regularizer):
    """
    Given a keras subclass-model, adds weight decay to all conv and dense layers.
    Inspired by: https://fantashit.com/is-it-the-same-adding-weight-decay-to-all-the-layers-including-input-and-output-layer-than-adding-the-weight-decay-term-to-the-cost-function/
    :param model: tf.keras.model to add the regularizer to
    :param kernel_regularizer: keras.regularizers
    :param bias_regularizer: keras.regularizers
    :return: the updated model
    """
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # this is e.g. in the ResNet where the model contains a sub-model
            for inner_layer in layer.layers:
                if hasattr(inner_layer, 'kernel_regularizer'):
                    inner_layer.kernel_regularizer = kernel_regularizer
                if hasattr(inner_layer, 'bias_regularizer'):
                    inner_layer.kernel_regularizer = bias_regularizer
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer
        if hasattr(layer, 'bias_regularizer'):
            layer.kernel_regularizer = bias_regularizer
    return model


MODELS = {
    "cifar10_cnn": cifar10_cnn,
    "cifar10_resnet50": cifar10_resnet50,
}
