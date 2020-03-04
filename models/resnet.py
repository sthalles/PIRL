import tensorflow as tf


class ResNetBase(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super(ResNetBase, self).__init__()
        self.base_model = tf.keras.applications.ResNet50(input_shape=input_shape,  # define the input shape
                                                           include_top=False,  # remove the classification layer
                                                           pooling='avg',
                                                           weights=None)  # use ImageNet pre-trained weights

        self.f = tf.keras.layers.Dense(units=output_dim, activation=None, name="head_f")
        self.g = tf.keras.layers.Dense(units=output_dim, activation=None, name="head_g")

    @tf.function
    def call(self, x, head, training=True):
        x = self.base_model(x, training=training)
        out = tf.cond(tf.equal(head, 'f'), lambda: self.f(x), lambda: self.g(x))
        return x, out
