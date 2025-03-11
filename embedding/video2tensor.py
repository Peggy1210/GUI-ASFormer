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


video_fmts = ["mp4", "m4v", "mkv", "webm", "mov", "avi", "wmv", "mpg", "flv"]

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


class ImgEmbedding:
    def __init__(self):
        pass
        
    def set_model(self,model_name:str):
        if model_name == "test":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = timm.create_model('swinv2_base_window12_192_22k', pretrained=True).to(self.device).eval()
            self.model_data_cofig = timm.data.resolve_model_data_config(self.model)
            self.spatial_attention = SpatialAttention(in_channels=1024).to(self.device)
            self.transforms = transforms.Compose([
                                    # image to num
                                    transforms.ToTensor(), # (C, H, W)
                                    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
                                ])

    def extract_features(self,framesmats):
        with torch.no_grad():
            output = self.model.forward_features(framesmats)
            output = self.spatial_attention(output.permute(0,3,1,2))
            return output.cpu()



def video2tensor(videos_folder: str, ft_folder: str, target_fps: int,batch_size:tuple,embedding:ImgEmbedding):
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



    
        



