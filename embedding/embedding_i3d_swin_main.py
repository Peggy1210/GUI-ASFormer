from huggingface_hub import hf_hub_download
from video2tensor import *
import shutil
import os
import sys 
# sys.path.append('/home/leilea/.local/bin')
sys.path.append(os.path.join(os.path.abspath(__file__),'..'))
    
def generate_features(file_path,start,end,video_batchsize,embedding_model):
  '''
  Args:
    file_path:str,path to retrieve the list of video names
    start:int, starting index
    end:int, ending index
    video_batchsize: int, # of videos in each zip file
    embedding_model:str, ["I3D","Swin"]

  '''
  ###################################################################

  if embedding_model == "Swin":
    embedding = SWin(batch_size=64)
    embedding.set_model(model_name="test")
    target_fps = 30
    batch_size = (256,256)
  elif embedding_model == "I3D":
    embedding = I3D()
    target_fps = 30
    batch_size = (224, 224)
  else:
    raise ValueError()

  # read video list
  video_list = []
  with open(file_path, "r") as f:
    video_list = f.readlines()

  print(f"Total number of videos: {len(video_list)}")

  # Process videos in batches
  patch_times= (end-start+1)//video_batchsize
  new_start, new_end = start, end

  for rd in range(patch_times):
    new_end = min(new_start+video_batchsize,end)
    print(f"Video Batch{rd}: Dowloading videos from index {new_start} to {new_end-1}. Number of downloads: {new_end-start}")

    # Create storage directory and download videos
    batch_video_list = [line.strip() for line in video_list[new_start:new_end]]
    data_path = f"data{new_start}-{new_end-1}"
    for video in batch_video_list:
        hf_hub_download(repo_id="shuaishuaicdp/GUI-World", filename=video, repo_type="dataset", local_dir=data_path)

    # Convert videos to tensors and extract features
    set_seed()
    videos_folder = data_path + '/' + batch_video_list[0].split("/")[0]
    ft_folder = f"features/{embedding_model}{new_start}-{new_end-1}"
    video2tensor(videos_folder=videos_folder,target_fps=target_fps,batch_size=batch_size,ft_folder=ft_folder,embedding=embedding,)

    # Clean up and compress
    # Delete caches
    torch.cuda.empty_cache()

    # This will zip the embeddings and delete the files
    shutil.make_archive(ft_folder, 'zip', ft_folder)
    shutil.rmtree(ft_folder)
    print(f'Features for video batch {rd} successfully zipped.')
    shutil.rmtree(data_path)

    new_start = new_end


file_path = "website_files.txt"
video_batchsize = 50
start = 100
end = 1000
embedding_model = "I3D"
generate_features(file_path,start,end,video_batchsize,embedding_model)

