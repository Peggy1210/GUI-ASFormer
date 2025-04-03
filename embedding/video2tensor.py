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
from pytorch_i3d import InceptionI3d
import json
import random

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


video_fmts = ["mp4", "m4v", "mkv", "webm", "mov", "avi", "wmv", "mpg", "flv"]
    
class I3DBlock(nn.Module):
    def __init__(self):
        super(I3DBlock, self).__init__()
        self.i3d = InceptionI3d(num_classes=400, in_channels=3)
        self.i3d.load_state_dict(torch.load('rgb_imagenet.pt'))
        self.i3d.eval()

        # Remove the classification head
        self.i3d.replace_logits(512)  # Remove final FC layer (I3D default is 400 classes)

        # Add a new 1x1 conv to expand from 1024 -> 2048
        self.conv3d = nn.Conv3d(1024, 2048, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        with torch.no_grad():
          x = self.i3d.extract_features(x)  # Extract I3D features
          x = self.conv3d(x)  # Expand to 2048 channels
          x = x.mean(dim=[2, 3, 4])  # Global average pooling (T, H, W)
        return x  # Shape: (Batch, 2048)

class I3D:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = transforms.Compose([
                            # image to num
                            transforms.ToTensor(), # (C, H, W)
                            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
                        ])
        self.i3d = I3DBlock()

    def extract_features(self, framesmats):
        with torch.no_grad(): 
            features = self.i3d(framesmats.permute(1, 0, 2, 3).unsqueeze(0))  # Extract deep features
            features = features.squeeze(0)

        return features
    

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # 1x1 convolution to generate attention weights

    def forward(self, x):
        """
        x: Features output by the Swin Transformer, shape (B, C, H, W)
        return: Weighted features, shape (B, C)
        """
        attention_weights = self.conv(x)  # (B, 1, H, W)

        # **Fix softmax dimension issue**
        attention_weights = attention_weights.flatten(2)  # Reshape to (B, 1, H*W)
        attention_weights = F.softmax(attention_weights, dim=-1)  # Apply softmax along the last dimension
        attention_weights = attention_weights.view_as(self.conv(x))  # Reshape back to (B, 1, H, W)

        # **Apply attention weights**
        attended_features = torch.sum(x * attention_weights, dim=(-2, -1))  # Sum over spatial dimensions, resulting in (B, C)

        return attended_features

class SWin:
    def __init__(self):
        pass
        
    def set_model(self,model_name:str):
        if model_name == "test":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = timm.create_model('swinv2_base_window8_256', pretrained=True).to(self.device).eval()
            self.model_data_cofig = timm.data.resolve_model_data_config(self.model)
            self.spatial_attention = SpatialAttention(in_channels=1024).to(self.device)
            self.transforms = transforms.Compose([
                                    # image to num
                                    transforms.ToTensor(), # (C, H, W)
                                    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
                                ])

    def extract_features(self,framesmats):
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(framesmats.shape[0]), desc="Processing Video"):
                frame = framesmats[i].unsqueeze(0).to(self.device)
                embedding = self.model.forward_features(frame)

                embed1, embed2 = embedding[:, :4, :4, :], embedding[:, 4:, 4:, :]
                embed1, embed2 = self.spatial_attention(embed1.permute(0, 3, 1, 2)), self.spatial_attention(embed2.permute(0, 3, 1, 2))

                combined_embedding = torch.cat([embed1, embed2], dim=1)
                all_embeddings.append(combined_embedding.cpu())

            final_embedding = torch.cat(all_embeddings, dim=0)
            final_embedding = final_embedding.permute(1, 0)
            print("final embedding shape:", final_embedding.shape)
            # output = self.model.forward_features(framesmats)
            # output = self.spatial_attention(output.permute(0,3,1,2))
            return final_embedding



def video2tensor(videos_folder: str, ft_folder: str, target_fps: int, batch_size:tuple, embedding):
    """
    Function to extract frames from videos using cv2

    Args:
        videos_folder:str. folder path where the raw videos are stored
        ft_folder:str. folder path where the video features are stored.
        timef:int. frame interperiod.
    """

    if not os.path.isdir(videos_folder):
        raise ValueError(f"folder doesn't exist {videos_folder}")

    if not os.path.isdir(ft_folder):
        os.makedirs(ft_folder)
    else:
        shutil.rmtree(ft_folder)
        os.makedirs(ft_folder)

    # extract frames
    videos_path = os.listdir(videos_folder)
    video_basic_info = []
    for idx,video_path in tqdm(enumerate(videos_path), total=len(videos_path)):
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
                frames.append(frame)
                
            c += 1
        cap.release()
        
        frames_tensor = torch.stack([embedding.transforms(frame) for frame in frames])
        # print(f"extracting features from {name}")
        features = embedding.extract_features(frames_tensor)
        # print(f"{name} feature extraction completed.")
        # print(f"shape:{features.shape}")
        np.save(os.path.join(ft_folder,f'{name}.npy'),features.numpy())
        
        cv2.destroyAllWindows()
        del features,frames_tensor
    
    with open(os.path.join(os.getcwd(),'../videos_basic_info.jsonl'),'w') as f:
        for line in video_basic_info:
            f.write(json.dumps(line)+'\n')

if __name__ == "main":
    abspath = os.path.join(os.getcwd(),'..')
    videos_folder = os.path.join(abspath,'videos')
    ft_folder = os.path.join(abspath,'features')
    target_fps = 30
    embedding = ImgEmbedding()
    embedding.set_model(model_name="test")
    batch_size = (192,192)
    
    video2tensor(videos_folder=videos_folder,target_fps=target_fps,batch_size=batch_size,ft_folder=ft_folder,embedding=embedding)



    
        



