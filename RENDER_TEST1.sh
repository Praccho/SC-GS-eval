#!/bin/bash

# job name
#SBATCH -J GS_EVAL

#
#SBATCH --account=ssrinath-gcondo

# partition
#SBATCH --partition=ssrinath-gcondo --gres=gpu:2 --gres-flags=enforce-binding

# ensures all allocated cores are on the same node
#SBATCH -N 1

# cpu cores
#SBATCH --ntasks-per-node=8

# memory per node
#SBATCH --mem=32G

# runtime
#SBATCH -t 24:00:00

# output
#SBATCH -o trender1.out

# error
#SBATCH -e trender1.err

# email notifiaction
# SBATCH --mail-type=ALL

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
module load cuda/11.8.0-lpttyok
conda activate pytorch-3d
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/turtle/ --model_path outputs/turtle/ --deform_type node --node_num 32 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --view_idx 1
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/turtle/ --model_path outputs/turtle/ --deform_type node --node_num 32 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode lbs --view_idx 1
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/turtle/ --model_path outputs/turtle/ --deform_type node --node_num 32 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode nodes --view_idx 1

CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/skater/ --model_path outputs/skater/ --deform_type node --node_num 32 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --view_idx 0
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/skater/ --model_path outputs/skater/ --deform_type node --node_num 32 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode lbs --view_idx 0
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/skater/ --model_path outputs/skater/ --deform_type node --node_num 32 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode nodes --view_idx 0

CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/boxes/ --model_path outputs/boxes/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --view_idx 1
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/boxes/ --model_path outputs/boxes/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode lbs --view_idx 1
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/boxes/ --model_path outputs/boxes/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode nodes --view_idx 1

CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/omnisim_custom/ --model_path outputs/omnisim_custom/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --view_idx 4
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/omnisim_custom/ --model_path outputs/omnisim_custom/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode lbs --view_idx 4
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/omnisim_custom/ --model_path outputs/omnisim_custom/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode nodes --view_idx 4