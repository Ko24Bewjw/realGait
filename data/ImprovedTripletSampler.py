# -*- coding: utf-8 -*-
# @Time : 12:26 2024/3/14
# @Author : Allen-wan
# @File : ImprovedTripletSampler.py
# @Software : PyCharm
import torch.utils.data as tordata
import random

class ImprovedTripletSampler(tordata.Sampler):
    def __init__(self, dataset_labels, batch_size):
        self.dataset_labels = dataset_labels
        self.p, self.k = batch_size
        self.labels_to_indices = {label: [] for label in set(self.dataset_labels)}
        for index, label in enumerate(self.dataset_labels):
            self.labels_to_indices[label].append(index)

        # Pre-calculate the length
        self.length = len(self.dataset_labels) // self.p * self.p

    def __iter__(self):
        for _ in range(len(self)):
            p_labels = random.sample(self.labels_to_indices.keys(), self.p)
            batch_indices = []
            for label in p_labels:
                indices = self.labels_to_indices[label]
                if len(indices) < self.k:
                    # If there are not enough samples, sample with replacement
                    choices = random.choices(indices, k=self.k)
                else:
                    # If there are enough samples, sample without replacement
                    choices = random.sample(indices, self.k)
                batch_indices.extend(choices)
            yield batch_indices

    def __len__(self):
        # Here, we define the number of batches per epoch
        # This could be adjusted based on actual training needs
        return self.length // self.p
