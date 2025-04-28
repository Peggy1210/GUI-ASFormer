from huggingface_hub import hf_hub_download
import shutil
import os 
import sys
from pathlib import Path
import torch

root_dir = os.path.dirname(__file__)
module_dir = str(Path(os.path.join(root_dir, "EVA/EVA-CLIP/rei")).resolve())
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from video2tensor_evaclip import EVACLIP,set_seed
from video2tensor_evaclip import video2tensor as video2tensor_evaclip


def generate_features_evaclip(file_path,start,end,video_batchsize):
  '''
  Args:
    file_path:str,path to retrieve the list of video names
    start:int, starting index
    end:int, ending index
    video_batchsize: int, # of videos in each zip file

  '''
  ###################################################################
  model_name = "EVA02-CLIP-L-14"
  pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
  linear_dim = (768,2048)
  target_fps = 30
  batch_size = (1024,1024)
  embedding = EVACLIP(model_name = model_name,pretrained = pretrained,linear_dim = linear_dim)

  # read video list
  video_list = []
  with open(file_path, "r") as f:
    video_list = f.readlines()

  print(f"Total number of videos: {len(video_list)}")

  # Process videos in batches
  patch_times= (end-start+1)//video_batchsize
  new_start,new_end = start,end

  for rd in range(patch_times):
    new_end = min(new_start+video_batchsize,end)
    print(f"Video Batch{rd}: Dowloading videos from index {new_start} to {new_end-1}. Number of downloads: {new_end-start}")

    # Create storage directory and download videos
    pvideo_list = [line.strip() for line in video_list[new_start:new_end]]
    data_path = os.path.join(root_dir,f"data{new_start}-{new_end-1}")
    for video in pvideo_list:
        hf_hub_download(repo_id="shuaishuaicdp/GUI-World", filename=video, repo_type="dataset", local_dir=data_path)

    # Convert videos to tensors and extract features
    set_seed()
    videos_folder = data_path + '/' + pvideo_list[0].split("/")[0]
    ft_folder = os.path.join(root_dir,f"features/eva{new_start}-{new_end-1}")
    video2tensor_evaclip(videos_folder=videos_folder,target_fps=target_fps,batch_size=batch_size,ft_folder=ft_folder,embedding=embedding)

    # Delete caches
    torch.cuda.empty_cache()

    # This will zip the embeddings and delete the files
    shutil.make_archive(ft_folder, 'zip', ft_folder)
    shutil.rmtree(ft_folder)
    print(f'Features for video batch {rd} successfully zipped.')
    shutil.rmtree(data_path)

    new_start = new_end


file_path = os.path.join(root_dir,"website_files.txt")
video_batchsize = 50
start = int(input('start index:'))
end = int(input('end index:'))
generate_features_evaclip(file_path,start,end,video_batchsize)