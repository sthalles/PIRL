import tensorflow as tf
import time

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed