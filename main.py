import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--data_dir', default="./data")
parser.add_argument('--dataset', default="website")
parser.add_argument('--split', default='1')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')

# For prediction model
parser.add_argument('--model_name', type=str, default=None)

# Embedding info
parser.add_argument('--features_low', type=str, default="swin")
parser.add_argument('--features_high', type=str, default="eva")
parser.add_argument('--input_dim_low', type=int, default=2048)
parser.add_argument('--input_dim_high', type=int, default=2048)

# Model Configuration
parser.add_argument('--num_epochs', type=int, default=120)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--sample_rate', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=10)
parser.add_argument('--num_f_maps', type=int, default=64)
parser.add_argument('--channel_mask_rate', type=float, default=0.1)

args = parser.parse_args()

vid_list_file = f"{args.data_dir}/{args.dataset}/splits/train.split{args.split}.bundle"
vid_list_file_tst = f"{args.data_dir}/{args.dataset}/splits/test.split{args.split}.bundle"
features_path_low = f"{args.data_dir}/{args.dataset}/features_{args.features_low}/"
features_path_high = f"{args.data_dir}/{args.dataset}/features_{args.features_high}/"
gt_path = f"{args.data_dir}/{args.dataset}/groundTruth/"
 
mapping_file = f"{args.data_dir}/{args.dataset}/mapping.txt"
 
model_dir = "./{}/".format(args.model_dir)+args.dataset+"/split_"+args.split

results_dir = "./{}/".format(args.result_dir)+args.dataset+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
 
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
num_classes = len(actions_dict)


trainer = Trainer(args.num_layers, 2, 2, args.num_f_maps, args.input_dim_low, num_classes, args.channel_mask_rate)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path_low, features_path_high, args.sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path_low, features_path_high, args.sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(model_dir, batch_gen, args.num_epochs, args.batch_size, args.lr, batch_gen_tst)

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path_low, features_path_high, args.sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    if args.model_name is None:
        model_name = "epoch-" + str(args.num_epoch) + ".model"

    trainer.predict(model_dir, results_dir, features_path_low, features_path_high, batch_gen_tst, model_name, actions_dict, args.sample_rate)

