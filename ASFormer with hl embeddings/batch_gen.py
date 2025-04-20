import numpy as np
import torch
import os
import random
import torch.nn.functional as F

def pad_to_4096(feature):
    C, T = feature.shape
    if C < 4096:
        padded = np.zeros((4096, T), dtype=feature.dtype)
        padded[:C, :] = feature
        return padded
    return feature

class BatchGenerator:
    def __init__(self, num_classes, actions_dict, gt_path, features_path_low, features_path_high, sample_rate):
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path_low = features_path_low
        self.features_path_high = features_path_high
        self.sample_rate = sample_rate

    def read_data(self, vid_list_file):
        self.list_of_examples = []
        with open(vid_list_file, 'r') as f:
            for line in f:
                self.list_of_examples.append(line.strip())
        self.index = 0
        random.shuffle(self.list_of_examples)

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def next_batch(self, batch_size, shuffle=True):
        batch_input_low, batch_input_high, batch_target, mask, vids = [], [], [], [], []
        batch_end = min(self.index + batch_size, len(self.list_of_examples))

        for i in range(self.index, batch_end):
            vid = self.list_of_examples[i].replace('.txt', '')
            vids.append(vid)

            features_low = np.load(os.path.join(self.features_path_low, vid + ".npy"))[:, ::self.sample_rate]
            # features_low = pad_to_4096(features_low)
            features_high = np.load(os.path.join(self.features_path_high, vid + ".npy"))[:, ::self.sample_rate]
            # features_high = pad_to_4096(features_high)
            gt = open(os.path.join(self.gt_path, vid + '.txt')).read().splitlines()
            label_seq = [self.actions_dict[label] for label in gt][::self.sample_rate]

            batch_input_low.append(torch.tensor(features_low, dtype=torch.float))
            batch_input_high.append(torch.tensor(features_high, dtype=torch.float))
            batch_target.append(torch.tensor(label_seq, dtype=torch.long).unsqueeze(0))
            mask.append(torch.ones(1, len(label_seq), dtype=torch.float))  # shape (1, T)

        max_len = max([x.shape[1] for x in batch_input_low])

        def pad_tensor(tensors):
            return torch.stack([F.pad(t, (0, max_len - t.shape[1])) for t in tensors])

        batch_input_low = pad_tensor(batch_input_low)      # shape (B, C, T)
        batch_input_high = pad_tensor(batch_input_high)    # shape (B, C, T)
        batch_target = pad_tensor(batch_target).squeeze(1) # shape (B, T)
        mask = pad_tensor(mask).repeat(1, self.num_classes, 1)  # shape (B, C, T)

        self.index = batch_end
        return batch_input_low, batch_input_high, batch_target, mask, vids

