import tensorflow as tf
from utils import transform, normalize
from models import build_model, f, g
from memory_bank import MemoryBank
import os
import pickle

tf.random.set_seed(99)

print(tf.test.is_gpu_available())

import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

indices = list(range(len(x_train)))

INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 1
EPOCHS = 1
N_NEGATIVES = 31

encoder = build_model(INPUT_SHAPE)
f_model = f(input_shape=(128,))
g_model = g(input_shape=(128,))

# create a dataset to initialize the Memory bank
dataset = tf.data.Dataset.from_tensor_slices((indices, x_train))
dataset = dataset.map(transform)
dataset = dataset.map(normalize)
dataset = dataset.repeat(1)
dataset = dataset.batch(1)

memory_bank = MemoryBank()

# # fill up the memory bank
# counter = 0
# for index, I, It in dataset:
#     v_i = encoder(I)
#     f_vi = np.squeeze(f_model(v_i))
#     memory_bank.add_or_update(index.numpy()[0], features=f_vi)
#     counter += 1
# print("Total iter:", counter)

assert len(memory_bank) == len(x_train.data)
# memory_bank.save_memory_bank()
print("Memory bank is filled!")

# recreate the dataset
dataset = tf.data.Dataset.from_tensor_slices((indices, x_train))
dataset = dataset.map(transform)
dataset = dataset.map(normalize)
dataset = dataset.shuffle(4096)
dataset = dataset.repeat(1)
dataset = dataset.batch(1)

cosine_similarity = tf.keras.losses.CosineSimilarity(reduction=tf.losses.Reduction.NONE, axis=-1)


def h(v_i, v_it, positive_index, T=0.07):
    assert v_i.shape == v_it.shape
    negatives = memory_bank.sample_negatives(positive_index, size=N_NEGATIVES)
    # print("negatives.shape:", negatives.shape)
    numerator = tf.math.exp(cosine_similarity(v_i, v_it) / T)
    # print("numerator.shape:", numerator.shape)

    negative_similarity = tf.math.exp(cosine_similarity(v_it, negatives) / T)
    return numerator / (numerator + tf.reduce_sum(negative_similarity))


def nce_loss(f_vi, g_vit, positive_index, eps=1e-15):
    assert f_vi.shape == g_vit.shape
    negatives = memory_bank.sample_negatives(positive_index, size=N_NEGATIVES)
    # print("negatives.shape:", negatives.shape)

    #  predicted input values of 0 and 1 are undefined (hence the clip by value)
    return -tf.math.log(tf.clip_by_value(h(f_vi, g_vit, positive_index), eps, 1 - eps)) - tf.reduce_sum(
        [tf.math.log(tf.clip_by_value(1 - h(g_vit, np.expand_dims(mi_prime, axis=0), positive_index), eps, 1 - eps)) for
         mi_prime in negatives])


def total_loss(mi, f_vi, g_vit, positive_index, lambda_=0.5):
    return lambda_ * nce_loss(mi, g_vit, positive_index) + (1 - lambda_) * nce_loss(mi, f_vi, positive_index)


optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

# fig, axs = plt.subplots(nrows=1, figsize=(24,4), ncols=BATCH_SIZE+1, constrained_layout=False)

MAX_ITER = 1000
counter = 0
for index, I, It in dataset:
    current_index = index.numpy()[0]

    with tf.GradientTape() as enc_tape, tf.GradientTape() as f_tape, tf.GradientTape() as g_tape:
        v_i = encoder(I)
        v_it = encoder(It)
        # print("----Feature vectors----")
        # print("v_i.shape:", v_i.shape)
        # print("v_it.shape:", v_it.shape)

        # compute the representations
        f_vi = f_model(v_i)
        g_vit = g_model(v_it)

        # update the representation in the memory bank
        memory_bank.add_or_update(current_index, np.squeeze(f_vi))

        # get the memory bank representation for the current image
        mi = memory_bank[current_index]
        # print("mi.shape:", mi.shape)

        loss = total_loss(mi, f_vi, g_vit, current_index)

    # compute grads w.r.t model parameters and update weights
    grads = enc_tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

    grads = f_tape.gradient(loss, f_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, f_model.trainable_variables))

    grads = g_tape.gradient(loss, g_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, g_model.trainable_variables))

    print("Loss:", loss)
    if counter == MAX_ITER:
        break
    counter += 1

encoder.save('./encoder.h5')