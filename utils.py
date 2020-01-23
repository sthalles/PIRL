import tensorflow as tf

def transform(index, image):
    random_angle = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
    # label 0 --> 0 degree
    # label 1 --> 90 degree
    # label 2 --> 180 degree
    # label 3 --> 270 degree
    label = random_angle
    image_transformed = tf.identity(image)
    if random_angle > 0:
        image_transformed = tf.image.rot90(image, k=random_angle)

    return index, image, image_transformed

def normalize(index, I, It):
    return index, tf.cast(I, tf.float32) / 255., tf.cast(It, tf.float32) / 255.