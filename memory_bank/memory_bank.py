import tensorflow as tf
import pickle
import os

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

    def init_memory_bank(self, batch_indices, batch_features):
        for idx, repr in zip(batch_indices, batch_features):
            self.memory_bank[idx].assign(repr)

    def update_memory_repr(self, batch_indices, batch_features):
        # perform batch update to the representations
        for idx, repr in zip(batch_indices, batch_features):
            self.memory_bank[idx].assign(self.weight * self.memory_bank[idx] + (1 - self.weight) * repr)

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