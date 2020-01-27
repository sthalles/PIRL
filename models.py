import tensorflow as tf
from utils import timeit

class CNN(tf.keras.Model):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                                            input_shape=input_shape)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.activation = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPooling2D(strides=2, pool_size=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

        self.f = tf.keras.layers.Dense(units=128, activation=None, name="head_f")
        self.g = tf.keras.layers.Dense(units=128, activation=None, name="head_g")

    # @timeit
    @tf.function
    def call(self, x, head, training=True):
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

        out = tf.cond(tf.equal(head, 'f'), lambda: self.f(x), lambda: self.g(x))
        # if head == 'f':
        #     out = self.f(x)
        # elif head == 'g':
        #     out = self.g(x)

        return x, out


def MobileNet(input_shape):
    # Create the base model from the pre-trained MobileNet V2
    base_model = tf.keras.applications.InceptionResNetV2(input_shape=input_shape,  # define the input shape
                                                         include_top=False,  # remove the classification layer
                                                         pooling='avg',
                                                         weights=None)  # use ImageNet pre-trained weights
    base_model.trainable = True
    return base_model