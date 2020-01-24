import tensorflow as tf
from utils import transform, normalize
from models import CNN
from memory_bank_tf import MemoryBankTf
import os
import pickle

tf.random.set_seed(99)

print(tf.test.is_gpu_available())
from keras.datasets import cifar10


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

indices = list(range(len(x_train)))

INPUT_SHAPE = (32, 32, 3)
EPOCHS = 1
N_NEGATIVES = 128

encoder = CNN(INPUT_SHAPE)
# _ = encoder(np.random.rand(1,32,32,3).astype(np.float32))
# encoder.load_weights('encoder.h5')

# create a dataset to initialize the Memory bank
memory_bank = MemoryBankTf(shape=(x_train.shape[0], 128), from_pickle=False)

dataset = tf.data.Dataset.from_tensor_slices((indices, x_train))
dataset = dataset.map(transform)
dataset = dataset.map(normalize)
dataset = dataset.repeat(EPOCHS)
dataset = dataset.batch(512)

# fill up the memory bank
counter = 0
for curr_indices, I, It in dataset:
    # print(np.min(I), np.max(I))
    # print(np.min(It), np.max(It))
    _, f_vi = encoder(I, head=tf.constant('f'))
    memory_bank.init_memory_bank(curr_indices, batch_features=f_vi)
    counter += 1

print("Total iter:", counter)

memory_bank.save_memory_bank()
print("Memory bank is filled!")