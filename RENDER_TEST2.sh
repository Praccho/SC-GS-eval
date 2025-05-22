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
#SBATCH -o render2.out

# error
#SBATCH -e render2.err

# email notifiaction
# SBATCH --mail-type=ALL

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
module load cuda/11.8.0-lpttyok
conda activate pytorch-3d
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/car/ --model_path outputs/car/ --deform_type node --node_num 32 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --view_idx 2
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/car/ --model_path outputs/car/ --deform_type node --node_num 32 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode lbs --view_idx 2
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/car/ --model_path outputs/car/ --deform_type node --node_num 32 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode nodes --view_idx 2

CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/restaurant_brunch/ --model_path outputs/restaurant_brunch/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --view_idx 2
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/restaurant_brunch/ --model_path outputs/restaurant_brunch/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode lbs --view_idx 2
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/restaurant_brunch/ --model_path outputs/restaurant_brunch/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode nodes --view_idx 2

CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/softball/ --model_path outputs/softball/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --view_idx 0
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/softball/ --model_path outputs/softball/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode lbs --view_idx 0
CUDA_VISIBLE_DEVICES=0 python render.py --source_path ../scratch/softball/ --model_path outputs/softball/ --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --iterations 51000 --mode time --render_mode nodes --view_idx 0