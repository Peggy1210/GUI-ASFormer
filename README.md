## Enviroment
Pytorch == 2.5.0, python == 3.11, CUDA=12.6

## Guideline
It's the modified version of ASFormer. This Dual-Encoder ASFormer has 1 deep encoder for features from low-resolution images and and 1 lightweighted encoder for features from high-resolution images.

## Data Location
Make sure your data is at the same location with the model. 

## Sample CLI Arguments 
python main.py \
  --mapping_path data/website/mapping.txt \
  --features_path_low data/website/features \
  --features_path_high data/website/features \
  --gt_path data/website/groundTruth \
  --vid_list_file data/website/splits/train.split1.bundle \
  --vid_list_file_tst data/website/splits/test.split1.bundle \
  --model_dir save/models \
  --results_dir save/results \
  --num_epochs 50 \
  --batch_size 1 \
  --sample_rate 2
