################################
# This python file will generate training labels from the annotaion file from GUI-World dataset.
# Please download the corresponding annotation file from:
# https://huggingface.co/datasets/shuaishuaicdp/GUI-World/tree/main/Annotation/train
# The output files will be stored under "<your_directory>/data/annotation".
# Please move this file to the directory with the same level of model to enable training. 
################################
from tqdm import tqdm
import pandas as pd
import json

data = []
with open('website.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

import pandas as pd
import os

LEGTH = len(data)
DATA_PATH = "data/annotation"
os.makedirs(DATA_PATH, exist_ok=True)
for i in tqdm(range(LEGTH)):
    df = []
    action_id = 1
    acc = 0
    video_id = data[i]["video_path"].split("/")[-1].split(".")[0]
    with open(f"{DATA_PATH}/{video_id}.txt", "w") as f:
        frames = sorted(data[i]["keyframes"], key=lambda x: x["frame"])
        for frame in frames:
            #   df.append({
            #       "video": video_id,
            #       "action_id": action_id,
            #       "sub_goal": frame["sub_goal"],
            #       "start": frame["frame"],

            #   })

            # The output format will be frame-wise action_ids separated by lines. For example:
            # /video1.txt
            # 0
            # 0
            # 0
            # 1
            # 1
            # 2
            # 3
            while acc < frame["frame"]:
                f.write(f"{action_id - 1}\n")
                acc += 1

            action_id += 1

        # TODO: Get the legnth of each videos and print the last action_id until the end
        f.write(f"{action_id - 1}\n")

#   df = pd.DataFrame(df)
#   df["end"] = df["start"].shift(-1).fillna(0).astype(int) - 1
#   df.to_csv(f"{DATA_PATH}/{video_id}.csv", index=False)