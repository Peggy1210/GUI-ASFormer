from eva_clip import create_model_and_transforms
from PIL import Image
import torch
from torch.nn.modules import linear
import torchvision.transforms as transforms
import torch.nn as nn
import os
import cv2
from tqdm import tqdm
from sys import exit
import shutil
import torchvision.transforms as transforms
import timm
import numpy as np 
import torch
import torch.nn.functional as F
import torch.nn as nn
import json 
import random 

video_fmts = ["mp4", "m4v", "mkv", "webm", "mov", "avi", "wmv", "mpg", "flv"]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class EVACLIP:
    def __init__(self,model_name,pretrained,linear_dim):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, self.transforms = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        self.model = model.to(self.device)
        ft_in,ft_out = linear_dim[0],linear_dim[1]
        self.linear_layer = nn.Linear(ft_in,ft_out,False).to(self.device)

    def extract_features(self,framesmats): # (T, C, H, W)
        all_embeddings = []
        with torch.no_grad(),torch.cuda.amp.autocast():
            for frame in framesmats:
                img = frame.unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(img)
                #     text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True) #1,768
                all_embeddings.append(image_features) #T,768
            
        all_embeddings = torch.cat(all_embeddings, dim=0) #T,768
        with torch.no_grad():
            all_embeddings_reshaped = self.linear_layer(all_embeddings.float())
        if self.device == "cuda": torch.cuda.empty_cache()
        return all_embeddings_reshaped.cpu()


def video2tensor(videos_folder: str, ft_folder: str, target_fps: int, batch_size:tuple, embedding, video_info=False, batch=False):
    """
    Function to extract frames from videos using cv2

    Args:
        videos_folder:str. folder path where the raw videos are stored
        ft_folder:str. folder path where the video features are stored.
        timef:int. frame interperiod.
    """

    if not os.path.isdir(videos_folder):
        raise ValueError(f"folder doesn't exist {videos_folder}")

    is_new = False
    if not os.path.isdir(ft_folder):
        os.makedirs(ft_folder)
        is_new = True
    

    # extract frames
    videos_path = os.listdir(videos_folder)
    
    if not is_new:
        ftembedded = [ele.split('.')[0] for ele in os.listdir(ft_folder)]
    
    video_basic_info = []
    for idx,video_path in tqdm(enumerate(videos_path), total=len(videos_path)):
        if not is_new:
            if video_path.split('.')[0] in ftembedded:
                continue 
            
        name, fmt = video_path.split(".")[0], video_path.split(".")[1]
        if fmt not in video_fmts:
            continue
        
        cap = cv2.VideoCapture(os.path.join(videos_folder, video_path))
        if not cap.isOpened():
            print(
                f"Cannot open camera.video_path:{os.path.join(videos_folder, video_path)}"
            )
            exit()
        
        fps,frames_cnt = cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = fps/min(target_fps,fps)
        video_basic_info.append({'id':idx,'name':name,'fps':fps,'duration/s':frames_cnt/fps})
  
        c = 0
        frames = []
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                break

            if c % frame_interval == 0:
                # Our operations on the frame come here
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, batch_size)
                frames.append(Image.fromarray(frame))
                
            c += 1

        cap.release()
        
        frames_tensor = torch.stack([embedding.transforms(frame) for frame in frames])
        # print(f"extracting features from {name}")
        features = embedding.extract_features(frames_tensor)
        # print(f"{name} feature extraction completed.")
        # print(f"shape:{features.shape}")
        np.save(os.path.join(ft_folder,f'{name}.npy'),features.numpy())
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        del features,frames_tensor
    
    if video_info:
        with open(os.path.join(os.getcwd(),'../videos_basic_info.jsonl'),'w') as f:
            for line in video_basic_info:
                f.write(json.dumps(line)+'\n')
