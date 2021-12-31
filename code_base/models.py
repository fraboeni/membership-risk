import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, UpSampling2D


class CIFAR10_CNN(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3))
        self.conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D((2, 2))
        self.dropout1 = Dropout(0.2)

        self.conv21 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling2D((2, 2))
        self.dropout2 = Dropout(0.2)

        self.conv31 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv32 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool3 = MaxPooling2D((2, 2))
        self.dropout3 = Dropout(0.2)

        self.flat = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout4 = Dropout(0.2)
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv11(inputs)
        x = self.conv12(x)
        x = self.pool1(x)
        if training:
            x = self.dropout1(x, training=training)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.pool2(x)
        if training:
            x = self.dropout2(x, training=training)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.pool3(x)
        if training:
            x = self.dropout3(x, training=training)

        x = self.flat(x)
        x = self.dense1(x)
        if training:
            x = self.dropout4(x, training=training)
        return self.dense2(x)


class CIFAR10_ResNet50(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                                       include_top=False,
                                                                       weights='imagenet')
        self.pool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(512, activation='relu')
        self.dense3 = Dense(10, activation="softmax")

    def call(self, inputs, training=False, base_trainable=False):
        x = UpSampling2D(size=(7, 7))(inputs)  # upsample 32, 32 to 224, 224 by multiplying with factor 7
        if not base_trainable:
            self.feature_extractor.trainable = False
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


# class FMNIST_CNN(tf.keras.Model):
#
#   def __init__(self):
#     super().__init__()
#     self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
#     self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
#     self.dropout = tf.keras.layers.Dropout(0.5)
#
#   def call(self, inputs , training=None, mask=None):
#     x = self.dense1(inputs)
#     if training:
#         x = self.dropout(x, training=training)
#
#     return self.dense2(x)




MODELS = {
    "cifar10_cnn": CIFAR10_CNN,
    "cifar10_resnet50": CIFAR10_ResNet50,
    # "fmnist": FMNIST_CNN,
}
