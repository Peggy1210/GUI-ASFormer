################################
# This python script will generate training labels from the annotaion file from GUI-World dataset.
# Please download the corresponding annotation file (.jsonl) from:
# https://huggingface.co/datasets/shuaishuaicdp/GUI-World/tree/main/Annotation/train
# The output files will be stored under `<your_directory>/data/annotation`.
################################
from tqdm import tqdm
import pandas as pd
import json
import os

DATASET = "website"
VIDEO_INFO = "video_len.json"

data = []
with open(f'{DATASET}.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

video_len = {}
with open(VIDEO_INFO, 'r') as f:
    video_len = json.load(f)
video_len_keys = list(video_len.keys())

LEGTH = len(data)
DATA_PATH = f"data/{DATASET}"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(f"{DATA_PATH}/groundTruth", exist_ok=True)
action_dict = { 'start': 0 }
action_id_inc = 0
for i in tqdm(range(LEGTH)):    
    video_id = data[i]["video_path"].split("/")[-1].split(".")[0]
    with open(f"{DATA_PATH}/groundTruth/{video_id}.txt", "w") as f:
        frames = sorted(data[i]["keyframes"], key=lambda x: x["frame"])

        # The output format will be frame-wise action_ids separated by lines. For example:
            # /video1.txt
            # start
            # start
            # start
            # action_1
            # action_1
            # action_2
            # action_3

        acc = 1
        action = 'start'
        for frame in frames:
            # Print out the previous action until the current frame
            while acc < frame["frame"]:
                f.write(f"{action}\n")
                acc += 1

            # Get the current action
            action = f'{frame["mouse"]}_{frame["keyboard"]}_{frame["keyboardOperation"] != ""}'

            # Add the action to the dictionary if it does not exist
            if action not in action_dict.keys():
                action_id_inc += 1
                action_dict[action] = action_id_inc

        # Print the last action_id until the end
        if video_id in video_len_keys:
            while acc <= video_len[video_id]:
                f.write(f"{action}\n")
                acc += 1
        else:
            f.write(f"{action}\n")

# Write mapping file
with open(f"{DATA_PATH}/mapping.txt", "w") as f:
    for action, id in action_dict.items():
        f.write(f"{id}, {action}\n")