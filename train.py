import time
import tensorflow as tf
import yaml
from data_aug.transforms import normalize, rotate_transform
from loss.nce_loss import nce_loss
from memory_bank.memory_bank import MemoryBankTf
from models.baseline import CNN
from models.resnet import ResNetBase
from utils import download_and_extract, read_all_images

tf.random.set_seed(99)
download_and_extract()

DATA_PATH = './data/stl10_binary/train_X.bin'
x_train = read_all_images(DATA_PATH)
input_shape = x_train.shape[1:]

print(tf.__version__)
print("Using GPU:", tf.test.is_gpu_available())

# x_train = x_train[:100]
print('x_train shape:', x_train.shape)

indices = list(range(len(x_train)))

config = yaml.load(open("stl10_config.yaml", "r"), Loader=yaml.FullLoader)

encoder = CNN(input_shape, config['out_dim'])
encoder = ResNetBase(input_shape, config['out_dim'])
# encoder.load_weights('encoder.h5')

# create a dataset to initialize the Memory bank
memory_bank = MemoryBankTf(shape=(x_train.shape[0], config['out_dim']), from_pickle=False)

# recreate the dataset
dataset = tf.data.Dataset.from_tensor_slices((indices, x_train))
dataset = dataset.map(rotate_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.repeat(config['epochs'])
dataset = dataset.shuffle(4096)
dataset = dataset.batch(config['batch_size'], drop_remainder=True)


def total_loss(mi, f_vi, g_vit, positive_index, lambda_):
    negatives = memory_bank.sample_negatives(positive_index, batch_size=mi.shape[0] * config['n_negatives'])

    term1 = lambda_ * nce_loss(mi, g_vit, negatives)
    term2 = (1 - lambda_) * nce_loss(mi, f_vi, negatives)

    tf.summary.scalar(name='nce_loss_term_1', data=tf.reduce_mean(term1), step=optimizer.iterations)
    tf.summary.scalar(name='nce_loss_term_2', data=tf.reduce_mean(term2), step=optimizer.iterations)

    del negatives  # seems to fix the problem of leakage
    return term1 + term2


optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

writer = tf.summary.create_file_writer('./logs/' + str(time.time()) + "/")


@tf.function
def train_step(batch_I, batch_It):
    with tf.GradientTape() as tape:
        v_i, f_vi = encoder(batch_I, head=tf.constant('f'), training=True)
        v_it, g_vit = encoder(batch_It, head=tf.constant('g'), training=True)

        tf.summary.histogram(name="v_i", data=v_i, step=optimizer.iterations)
        tf.summary.histogram(name="v_it", data=v_it, step=optimizer.iterations)
        tf.summary.histogram(name="f_vi", data=f_vi, step=optimizer.iterations)
        tf.summary.histogram(name="g_vit", data=g_vit, step=optimizer.iterations)

        # get the memory bank representation for the current image
        mi = memory_bank.sample_by_indices(curr_indices)
        tf.summary.histogram(name="mi", data=mi, step=optimizer.iterations)
        assert mi.shape == (config['batch_size'], config['out_dim']), "Shape does not match --> " + str(mi.shape)

        loss = tf.reduce_mean(total_loss(mi, f_vi, g_vit, curr_indices, lambda_=config['lambda']))
        tf.summary.scalar('loss', loss, step=optimizer.iterations)

    # update the representation in the memory bank
    memory_bank.update_memory_repr(curr_indices, f_vi)

    # compute grads w.r.t model parameters and update weights
    grads = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables))


with writer.as_default():
    for curr_indices, I, It in dataset:
        start = time.time()
        train_step(I, It)

        end = time.time()
        print("Time/batch:", (end-start) * 1000, "ms")

encoder.save_weights('./checkpoints/encoder.h5')
# memory_bank.save_memory_bank()
