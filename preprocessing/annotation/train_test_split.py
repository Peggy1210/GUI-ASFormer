################################
# This python script splits the training and testing data.
# Change the `DATA_PATH` to your dataset directory, and specify 
# the number of splits and `TRAIN_TEST_SPLIT`.
# The script will generates corresponding numbers of train and test bundles
# stored in `<DATA_PATH>/data/<DATASET>/splits`.
################################
import os
import random

SEED = 123
random.seed(SEED)

DATASET = "website"
DATA_PATH = "."
NUM_SPLIT = 2
TRAIN_TEST_SPLIT = 0.8

dataset = list(os.listdir(f"{DATA_PATH}/data/{DATASET}/features"))
# Check if corresponding annotation file exists
dataset = [d for d in dataset if d.split(".")[1] == "npy" and os.path.isfile(f"{DATA_PATH}/data/{DATASET}/groundTruth/{d.split('.')[0]}.txt")]

# Split dataset
random.shuffle(dataset)
os.makedirs(f"{DATA_PATH}/data/{DATASET}/splits", exist_ok=True)
DATASET_LEN = len(dataset)
SPLIT_LEN = DATASET_LEN // NUM_SPLIT

for split_idx in range(NUM_SPLIT):
    start_idx = split_idx * SPLIT_LEN
    split_dataset = dataset[start_idx:start_idx + SPLIT_LEN]
    TRAIN_IDX = int(TRAIN_TEST_SPLIT * len(split_dataset))
    train, test = split_dataset[:TRAIN_IDX], split_dataset[TRAIN_IDX:]
    
    with open(f"{DATA_PATH}/data/{DATASET}/splits/train.split{split_idx + 1}.bundle", "w") as f:
        for file in train:
            f.write(f"{file.split('.')[0]}.txt\n")
    with open(f"{DATA_PATH}/data/{DATASET}/splits/test.split{split_idx + 1}.bundle", "w") as f:
        for file in test:
            f.write(f"{file.split('.')[0]}.txt\n")