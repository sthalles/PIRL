import numpy as np
import os
import pickle
import tensorflow as tf

np.random.seed(99)
import time


class MemoryBank:
    def __init__(self, shape, weight=0.5):
        self.weight = weight
        self.shape = shape
        self.filename = "./memory_bank.pkl"
        self.memory_bank = self._init_memory_bank_from_file()

    def _init_memory_bank_from_file(self):
        if os.path.isfile(self.filename):
            print("Memory bank loaded from file.")
            return pickle.load(open(self.filename, "rb"))
        else:
            print("Memory bank empty.")
            stdev = 1. / np.sqrt(self.shape[1] / 3)
            memory = np.random.rand(self.shape[0], self.shape[1]).astype(np.float32)
            memory *= 2 * stdev
            memory -= stdev
            return memory

    def save_memory_bank(self):
        if not os.path.isfile(self.filename):
            with open(self.filename, "wb") as f:
                pickle.dump(self.memory_bank, f)
                print("Memory bank saved as:", self.filename)
        else:
            print("Memory bank empty.")

    def init_memory_bank(self, batch_indices, batch_features):
        self.memory_bank[batch_indices] = batch_features

    def update_memory_repr(self, batch_indices, batch_features):

        for i, repr in zip(batch_indices, batch_features):
            assert len(repr.shape) == 1, "The memory bank accepts only single dim vectors -- Shape:" + str(
                repr.shape)
            # update the representations with an exponential moving average
            features_updated = self.weight * self.memory_bank[i] + (1 - self.weight) * repr
            self.memory_bank[i] = features_updated

    def __len__(self):
        return len(self.memory_bank)

    def __getitem__(self, idx):
        return np.expand_dims(self.memory_bank[idx], axis=0)

    def sample_negatives(self, positive_indices, batch_size):
        # returns a batch of representations from the memory bank
        # the [index] parameter value excludes the corresponding image
        # from the output batch
        # start = time.time()

        p = np.ones(self.memory_bank.shape[0])
        p[positive_indices] = 0
        p = p / np.sum(p)
        # assert np.sum(p) == 1.0, "The probabilities must sum up to 1. Received: " + str(np.sum(p))

        candidate_negative_samples = np.random.choice(list(range(self.memory_bank.shape[0])), p=p, size=batch_size)

        while len(np.intersect1d(candidate_negative_samples, positive_indices)) > 0:
            candidate_negative_samples = np.random.choice(list(range(self.memory_bank.shape[0])), p=p, size=batch_size)
            print("retrying...")

        # do not use np.array([list comprehention])!
        # the perfomance slows down exponentially
        negatives = self.memory_bank[candidate_negative_samples]
        negatives = tf.convert_to_tensor(negatives, dtype=tf.float32)
        # end = time.time()
        # print("Time:", end-start)
        return negatives

    def sample_by_indices(self, batch_indices):
        # returns a batch of representations from the memory bank by ids
        # start = time.time()
        batch = self.memory_bank[batch_indices]
        batch = tf.convert_to_tensor(batch)
        # end = time.time()
        # print("Time:", end-start)
        return batch
