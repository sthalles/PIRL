from tensorflow.keras.layers import Conv2D, Input, GlobalAveragePooling2D, LeakyReLU, Flatten, Dense, Reshape, \
    MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation, Input, Dropout, SpatialDropout2D
from tensorflow.keras import Model, Sequential


def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=3, strides=1, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(GlobalAveragePooling2D())
    return model

def f(input_shape):
    model = Sequential()
    model.add(Dense(units=128))
    return model

def g(input_shape):
    model = Sequential()
    model.add(Dense(units=128))
    return model