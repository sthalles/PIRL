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
x_train = x_train[:5000]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

indices = list(range(len(x_train)))

INPUT_SHAPE = (32, 32, 3)
EPOCHS = 1
N_NEGATIVES = 128

encoder = build_model(INPUT_SHAPE)
f_model = f(input_shape=(128,))
g_model = g(input_shape=(128,))

# create a dataset to initialize the Memory bank
memory_bank = MemoryBank(shape=(x_train.shape[0], 128))

dataset = tf.data.Dataset.from_tensor_slices((indices, x_train))
dataset = dataset.map(transform)
dataset = dataset.map(normalize)
dataset = dataset.repeat(EPOCHS)
dataset = dataset.batch(32)
# fill up the memory bank
counter = 0
for curr_indices, I, It in dataset:
    v_i = encoder(I)
    f_vi = f_model(v_i)
    memory_bank.init_memory_bank(curr_indices.numpy(), batch_features=f_vi)
    counter += 1
print("Total iter:", counter)

assert len(memory_bank) == x_train.shape[0]
memory_bank.save_memory_bank()
print("Memory bank is filled!")