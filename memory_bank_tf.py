import tensorflow as tf
import pickle
from utils import timeit
import os

tf.random.set_seed(99)


class MemoryBankTf:
    def __init__(self, shape, weight=0.5, from_pickle=False):
        self.weight = weight
        self.shape = shape
        self.filename = "./memory_bank.pkl"

        if from_pickle:
            memory_initializer = tf.constant(pickle.load(open(self.filename, "rb")))
        else:
            memory_initializer = tf.random.truncated_normal(shape)

        self.memory_bank = tf.Variable(memory_initializer, trainable=False)

    def save_memory_bank(self):
        if not os.path.isfile(self.filename):
            with open(self.filename, "wb") as f:
                pickle.dump(self.memory_bank.numpy(), f)
                print("Memory bank saved as:", self.filename)
        else:
            print("Memory bank empty.")

    @timeit
    def init_memory_bank(self, batch_indices, batch_features):
        for idx, repr in zip(batch_indices, batch_features):
            self.memory_bank[idx].assign(repr)

    def update_memory_repr(self, batch_indices, batch_features):
        # get the corresponding embeddings
        embeddigs = tf.nn.embedding_lookup(self.memory_bank, batch_indices)

        # perform batch update to the representations
        features_updated_ = self.weight * embeddigs + (1 - self.weight) * batch_features

        # update
        for idx, repr in zip(batch_indices, features_updated_):
            self.memory_bank[idx].assign(repr)

    def sample_negatives(self, positive_indices, batch_size):
        positive_indices = tf.expand_dims(positive_indices, axis=1)
        updates = tf.zeros(positive_indices.shape[0], dtype=tf.int32)

        mask = tf.ones([self.shape[0]], dtype=tf.int32)
        mask = tf.tensor_scatter_nd_update(mask, positive_indices, updates)

        p = tf.ones(self.shape[0])
        p = p * tf.cast(mask, tf.float32)
        p = p / tf.reduce_sum(p)

        candidate_negative_indices = tf.random.categorical(tf.math.log(tf.reshape(p, (1, -1))),
                                                           batch_size)  # note log-prob
        embeddings = tf.nn.embedding_lookup(self.memory_bank, tf.squeeze(candidate_negative_indices))
        return embeddings

    def sample_by_indices(self, batch_indices):
        return tf.nn.embedding_lookup(self.memory_bank, batch_indices)

# memory_bank = MemoryBankTf(shape=[10, 3])
# assert memory_bank.memory_bank.shape == [10, 3], "Failed! Received:" + str(len(memory_bank))
#
# negatives = memory_bank.sample_negatives(positive_indices=[1, 2, 3, 4, 5, 6, 7, 8, 9])
#
# negatives = memory_bank.sample_by_indices(batch_indices=[8])
#
# memory_bank.update_memory_repr(tf.convert_to_tensor([4, 1]),
#                                tf.convert_to_tensor([[0, 0, 0], [0, 0, 0]], dtype=tf.float32))
#
# negatives = memory_bank.sample_by_indices(batch_indices=[4])
# print(negatives)
