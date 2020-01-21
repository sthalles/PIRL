import numpy as np
import os
import pickle
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

    def add_or_update(self, index, features):
        assert len(features.shape) == 1, "The memory bank accepts only single dim vectors -- Shape:" + str(
            features.shape)
        if index not in self.memory_bank:
            self.memory_bank[index] = features
        else:
            # update the representations with an exponential moving average
            features_updated = self.weight * self.memory_bank[index] + (1 - self.weight) * features
            self.memory_bank[index] = features_updated

    def __len__(self):
        return len(self.memory_bank)

    def __getitem__(self, idx):
        return np.expand_dims(self.memory_bank[idx], axis=0)

    def sample_negatives(self, index, size):
        # returns a batch of representations from the memory bank
        # the [index] parameter value excludes the corresponding image
        # from the output batch
        candidate_negative_samples = [index]
        while index in candidate_negative_samples:
            candidate_negative_samples = np.random.choice(list(self.memory_bank.keys()), size)

        return np.array([self.memory_bank[i] for i in candidate_negative_samples])