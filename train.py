import tensorflow as tf

print(tf.__version__)
from utils import transform, normalize, timeit
from models import MobileNet, CNN
# from memory_bank import MemoryBank
from memory_bank_tf import MemoryBankTf
import time
import pickle

tf.random.set_seed(99)
print(tf.test.is_gpu_available())
from tensorflow.keras.datasets import cifar10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train[:10000]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

indices = list(range(len(x_train)))

INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 32
EPOCHS = 50
N_NEGATIVES = 256+128

encoder = CNN(INPUT_SHAPE)
# _ = encoder(np.random.rand(1,32,32,3))
# encoder.load_weights('encoder.h5')

# create a dataset to initialize the Memory bank
memory_bank = MemoryBankTf(shape=(x_train.shape[0], 128), from_pickle=False)

# recreate the dataset
dataset = tf.data.Dataset.from_tensor_slices((indices, x_train))
dataset = dataset.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.repeat(EPOCHS)
dataset = dataset.shuffle(4096)
dataset = dataset.batch(BATCH_SIZE)

cosine_similarity = tf.keras.losses.CosineSimilarity(reduction=tf.losses.Reduction.NONE, axis=-1)


def h(v_i, v_it, negatives, T=0.07):
    assert v_i.shape == v_it.shape, "Shapes do not match" + str(v_i.shape) + ' != ' + str(v_it.shape)

    # print("negatives.shape:", negatives.shape)
    similarity = cosine_similarity(v_i, v_it)
    numerator = tf.math.exp(similarity / T)
    # print("numerator.shape:", numerator.shape)

    v_it = tf.expand_dims(v_it, axis=1)

    similarity = cosine_similarity(v_it, tf.expand_dims(negatives, axis=0))
    negative_similarity = tf.math.exp(similarity / T)
    negative_similarity = tf.reduce_sum(negative_similarity, axis=1)
    return numerator / (numerator + negative_similarity)


# @timeit
def nce_loss(f_vi, g_vit, negatives, eps=1e-15):
    assert f_vi.shape == g_vit.shape, "Shapes do not match" + str(f_vi.shape) + ' != ' + str(g_vit.shape)
    #  predicted input values of 0 and 1 are undefined (hence the clip by value)

    return - tf.math.log(h(f_vi, g_vit, negatives)) - tf.math.log(1 - h(g_vit, negatives[:BATCH_SIZE, :], negatives))


def total_loss(mi, f_vi, g_vit, positive_index, lambda_=0.5):
    negatives = memory_bank.sample_negatives(positive_index, batch_size=mi.shape[0] * N_NEGATIVES)

    term1 = lambda_ * nce_loss(mi, g_vit, negatives)
    term2 = (1 - lambda_) * nce_loss(mi, f_vi, negatives)

    tf.summary.scalar(name='nce_loss_term_1', data=tf.reduce_mean(term1), step=optimizer.iterations)
    tf.summary.scalar(name='nce_loss_term_2', data=tf.reduce_mean(term2), step=optimizer.iterations)

    # seems to fix the problem of leakage
    del negatives
    return term1 + term2


optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

# fig, axs = plt.subplots(nrows=1, figsize=(24,4), ncols=BATCH_SIZE+1, constrained_layout=False)

writer = tf.summary.create_file_writer('./logs/' + str(time.time()) + "/")

counter = 0

with writer.as_default():
    for curr_indices, I, It in dataset:
        # start = time.time()
        with tf.GradientTape() as tape:
            v_i, f_vi = encoder(I, head=tf.constant('f'), training=True)
            v_it, g_vit = encoder(It, head=tf.constant('g'), training=True)

            tf.summary.histogram(name="v_i", data=v_i, step=optimizer.iterations)
            tf.summary.histogram(name="v_it", data=v_it, step=optimizer.iterations)
            tf.summary.histogram(name="f_vi", data=f_vi, step=optimizer.iterations)
            tf.summary.histogram(name="g_vit", data=g_vit, step=optimizer.iterations)

            # get the memory bank representation for the current image
            mi = memory_bank.sample_by_indices(curr_indices)
            tf.summary.histogram(name="mi", data=mi, step=optimizer.iterations)
            # assert mi.shape == (1,128), "Shape does not match --> " + str(mi.shape) + "Index:" + str(current_indices)

            loss = tf.reduce_mean(total_loss(mi, f_vi, g_vit, curr_indices))
            tf.summary.scalar('loss', loss, step=optimizer.iterations)

        # update the representation in the memory bank
        memory_bank.update_memory_repr(curr_indices, f_vi)

        # compute grads w.r.t model parameters and update weights
        grads = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

        # end = time.time()
        # print("Loss:", loss.numpy(), "Time/batch:", (end-start) * 1000, "ms")

# encoder.save('saved_model/my_model')
encoder.save_weights('encoder.h5')
# memory_bank.save_memory_bank()
