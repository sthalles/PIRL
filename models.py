import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same',
                                            input_shape=input_shape)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPooling2D(strides=2, pool_size=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x, training):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.global_pool(x)
        return x


class f(tf.keras.Model):
    def __init__(self):
        super(f, self).__init__()
        self.dense = tf.keras.layers.Dense(units=128, activation=None)

    def call(self, x):
        x = self.dense(x)
        return x


class g(tf.keras.Model):

    def __init__(self):
        super(g, self).__init__()
        self.dense = tf.keras.layers.Dense(units=128, activation=None)

    def call(self, x):
        x = self.dense(x)
        return x


def MobileNet(input_shape):

    # Create the base model from the pre-trained MobileNet V2
    base_model = tf.keras.applications.InceptionResNetV2(input_shape=input_shape,  # define the input shape
                                                         include_top=False,  # remove the classification layer
                                                         pooling='avg',
                                                         weights=None)  # use ImageNet pre-trained weights
    base_model.trainable = True
    return base_model

# def build_model(input_shape):
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', input_shape=input_shape))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation('relu'))
#     model.add(layers.MaxPooling2D(pool_size=2))
#
#     model.add(layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation('relu'))
#     model.add(layers.MaxPooling2D(pool_size=2))
#
#     model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation('relu'))
#     model.add(layers.MaxPooling2D(pool_size=2))
#
#     model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation('relu'))
#     model.add(layers.MaxPooling2D(pool_size=2))
#
#     model.add(layers.GlobalAveragePooling2D())
#     return model

# def f():
#     model = Sequential()
#     model.add(Dense(units=128))
#     return model
#
# def g():
#     model = Sequential()
#     model.add(Dense(units=128))
#     return model
