'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
from grid_sampler import GridSampler

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path_low, features_path_high, sample_rate):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path_low = features_path_low
        self.features_path_high = features_path_high
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        self.gts = [self.gt_path + vid for vid in self.list_of_examples]
        self.features_low = [self.features_path_low + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        self.features_high = [self.features_path_high + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        self.my_shuffle()

    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        
        combined = list(zip(self.features_low, self.features_high))
        random.shuffle(combined)
        self.features_low, self.features_high = zip(*combined)
        self.features_low = list(self.features_low)
        self.features_high = list(self.features_high)


    def warp_video(self, batch_input_tensor_low, batch_input_tensor_high, batch_target_tensor):
        '''
        :param batch_input_tensor_low: (bs, C_in_low, L_in)
        :param batch_input_tensor_high: (bs, C_in_high, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor_low.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor_low = self.timewarp_layer(batch_input_tensor_low, grid, mode='bilinear')
        warped_batch_input_tensor_high = self.timewarp_layer(batch_input_tensor_high, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape

        return warped_batch_input_tensor_low, warped_batch_input_tensor_high, warped_batch_target_tensor

    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features_low += bg.features_low
        self.features_high += bg.features_high

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))


    def next_batch(self, batch_size, if_warp=False): # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features_low = self.features_low[self.index:self.index + batch_size]
        batch_features_high = self.features_high[self.index:self.index + batch_size]

        self.index += batch_size

        batch_input_low = []
        batch_input_high = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features_low = np.load(batch_features_low[idx])
            features_high = np.load(batch_features_high[idx])
            file_ptr = open(batch_gts[idx], 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features_low)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            feature_low = features_low[:, ::self.sample_rate]
            feature_high = features_high[:, ::self.sample_rate]
            target = classes[::self.sample_rate]
            batch_input_low.append(feature_low)
            batch_input_high.append(feature_high)
            batch_target.append(target)

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor_low = torch.zeros(len(batch_input_low), np.shape(batch_input_low[0])[0], max(length_of_sequences), dtype=torch.float)  # bs, C_in_low, L_in
        batch_input_tensor_high = torch.zeros(len(batch_input_high), np.shape(batch_input_high[0])[0], max(length_of_sequences), dtype=torch.float)  # bs, C_in_high, L_in
        batch_target_tensor = torch.ones(len(batch_input_low), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input_low), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input_low)):
            if if_warp:
                warped_input_low, warped_input_high, warped_target = self.warp_video(torch.from_numpy(batch_input_low[i]).unsqueeze(0), torch.from_numpy(batch_input_high[i]).unsqueeze(0),  torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor_low[i, :, :np.shape(batch_input_low[i])[1]], batch_target_tensor[i, :np.shape(batch_target[i])[0]] = warped_input_low.squeeze(0), warped_target.squeeze(0)
                batch_input_tensor_high[i, :, :np.shape(batch_input_high[i])[1]], batch_target_tensor[i, :np.shape(batch_target[i])[0]] = warped_input_high.squeeze(0), warped_target.squeeze(0)
            else:
                batch_input_tensor_low[i, :, :np.shape(batch_input_low[i])[1]] = torch.from_numpy(batch_input_low[i])
                batch_input_tensor_high[i, :, :np.shape(batch_input_high[i])[1]] = torch.from_numpy(batch_input_high[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor_low, batch_input_tensor_high, batch_target_tensor, mask, batch


if __name__ == '__main__':
    pass