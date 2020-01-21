import numpy as np
import os
import pickle
import tensorflow as tf
np.random.seed(99)

class MemoryBank:
    def __init__(self, weight=0.5):
        self.weight = weight
        self.filename = "./memory_bank.pkl"
        self.memory_bank = self._init_memory_bank_from_file()

    def _init_memory_bank_from_file(self):
        if os.path.isfile(self.filename):
            print("Memory bank loaded from file.")
            return pickle.load(open(self.filename, "rb"))
        else:
            print("Memory bank empty.")
            return {}

    def save_memory_bank(self):
        if not os.path.isfile(self.filename):
            with open(self.filename, "wb") as f:
                pickle.dump(self.memory_bank, f)
                print("Memory bank saved as:", self.filename)
        else:
            print("Memory bank empty.")

    def add_or_update(self, batch_indices, batch_features):

        for i, repr in zip(batch_indices, batch_features):
            assert len(repr.shape) == 1, "The memory bank accepts only single dim vectors -- Shape:" + str(
                repr.shape)
            if i not in self.memory_bank:
                self.memory_bank[i] = repr
            else:
                # update the representations with an exponential moving average
                features_updated = self.weight * self.memory_bank[i] + (1 - self.weight) * repr
                self.memory_bank[i] = features_updated

    def __len__(self):
        return len(self.memory_bank)

    def __getitem__(self, idx):
        return np.expand_dims(self.memory_bank[idx], axis=0)

    def sample_negatives(self, positive_indices, size):
        # returns a batch of representations from the memory bank
        # the [index] parameter value excludes the corresponding image
        # from the output batch
        candidate_negative_samples = positive_indices.copy()
        while positive_indices in candidate_negative_samples:
            candidate_negative_samples = np.random.choice(list(self.memory_bank.keys()), size)

        return np.array([self.memory_bank[i] for i in candidate_negative_samples])

    def sample_by_indices(self, batch_indices):
        # returns a batch of representations from the memory bank by ids
        return np.array([self.memory_bank[i] for i in batch_indices])