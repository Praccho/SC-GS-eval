#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical
from utils.sh_utils import RGB2SH
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
from pytorch_msssim import ms_ssim
from piq import LPIPS
lpips = LPIPS()
from utils.image_utils import ssim as ssim_func
from utils.image_utils import psnr, lpips, alex_lpips

import matplotlib.colors as mcolors

def get_contrastive_colors(N, saturation=0.7, value=0.9):
    hues = np.linspace(0, 1, N, endpoint=False)
    hsv_colors = np.stack([hues, np.full(N, saturation), np.full(N, value)], axis=1)
    rgb_colors = np.array([mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors])
    return torch.from_numpy(rgb_colors)

def get_chaerin_colors(N, saturation=0.7, value=0.9):
    base = np.array([
        0.5020,         0,         0,
            0,    0.5020,         0,
        0.5020,    0.5020,         0,
            0,         0,    0.5020,
        0.5020,         0,    0.5020,
            0,    0.5020,    0.5020,
        0.7020,    0.7020,    0.7020,
        0.2510,         0,         0,
        0.7529,         0,         0,
        0.2510,    0.5020,         0,
        0.7529,    0.5020,         0,
        0.2510,         0,    0.5020,
        0.7529,         0,    0.5020,
        0.2510,    0.5020,    0.5020,
        0.7529,    0.5020,    0.5020,
            0,    0.2510,         0,
        0.5020,    0.2510,         0,
            0,    0.7529,         0,
        0.5020,    0.7529,         0,
            0,    0.2510,    0.5020,
        0.5020,    0.2510,    0.5020,
            0,    0.7529,    0.5020,
        0.5020,    0.7529,    0.5020,
        0.2510,    0.2510,         0,
        0.7529,    0.2510,         0,
        0.2510,    0.7529,         0,
        0.7529,    0.7529,         0,
        0.2510,    0.2510,    0.5020,
        0.7529,    0.2510,    0.5020,
        0.2510,    0.7529,    0.5020,
        0.7529,    0.7529,    0.5020,
            0,         0,    0.2510,
        0.5020,         0,    0.2510,
            0,    0.5020,    0.2510,
        0.5020,    0.5020,    0.2510,
            0,         0,    0.7529,
        0.5020,         0,    0.7529,
            0,    0.5020,    0.7529,
        0.5020,    0.5020,    0.7529,
        0.2510,         0,    0.2510,
        0.7529,         0,    0.2510,
        0.2510,    0.5020,    0.2510,
        0.7529,    0.5020,    0.2510,
        0.2510,         0,    0.7529,
        0.7529,         0,    0.7529,
        0.2510,    0.5020,    0.7529,
        0.7529,    0.5020,    0.7529,
            0,    0.2510,    0.2510,
        0.5020,    0.2510,    0.2510,
            0,    0.7529,    0.2510,
        0.5020,    0.7529,    0.2510,
            0,    0.2510,    0.7529,
        0.5020,    0.2510,    0.7529,
            0,    0.7529,    0.7529,
        0.5020,    0.7529,    0.7529,
        0.7529,    0.2510,    0.2510,
        0.2510,    0.7529,    0.2510,
        0.7529,    0.7529,    0.2510,
        0.2510,    0.2510,    0.7529,
        0.7529,    0.2510,    0.7529,
        0.2510,    0.7529,    0.7529,
        0.1255,         0,         0,
        0.6275,         0,         0,
        0.1255,    0.5020,         0,
        0.6275,    0.5020,         0,
        0.1255,         0,    0.5020,
        0.6275,         0,    0.5020,
        0.1255,    0.5020,    0.5020,
        0.6275,    0.5020,    0.5020,
        0.3765,         0,         0,
        0.8784,         0,         0,
        0.3765,    0.5020,         0,
        0.8784,    0.5020,         0,
        0.3765,         0,    0.5020,
        0.8784,         0,    0.5020,
        0.3765,    0.5020,    0.5020,
        0.8784,    0.5020,    0.5020,
        0.1255,    0.2510,         0,
        0.6275,    0.2510,         0,
        0.1255,    0.7529,         0,
        0.6275,    0.7529,         0,
        0.1255,    0.2510,    0.5020,
        0.6275,    0.2510,    0.5020,
        0.1255,    0.7529,    0.5020,
        0.6275,    0.7529,    0.5020,
        0.3765,    0.2510,         0,
        0.8784,    0.2510,         0,
        0.3765,    0.7529,         0,
        0.8784,    0.7529,         0,
        0.3765,    0.2510,    0.5020,
        0.8784,    0.2510,    0.5020,
        0.3765,    0.7529,    0.5020,
        0.8784,    0.7529,    0.5020,
        0.1255,         0,    0.2510,
        0.6275,         0,    0.2510,
        0.1255,    0.5020,    0.2510,
        0.6275,    0.5020,    0.2510,
        0.1255,         0,    0.7529,
        0.6275,         0,    0.7529,
        0.1255,    0.5020,    0.7529,
        0.6275,    0.5020,    0.7529,
        0.3765,         0,    0.2510,
        0.8784,         0,    0.2510,
        0.3765,    0.5020,    0.2510,
        0.8784,    0.5020,    0.2510,
        0.3765,         0,    0.7529,
        0.8784,         0,    0.7529,
        0.3765,    0.5020,    0.7529,
        0.8784,    0.5020,    0.7529,
        0.1255,    0.2510,    0.2510,
        0.6275,    0.2510,    0.2510,
        0.1255,    0.7529,    0.2510,
        0.6275,    0.7529,    0.2510,
        0.1255,    0.2510,    0.7529,
        0.6275,    0.2510,    0.7529,
        0.1255,    0.7529,    0.7529,
        0.6275,    0.7529,    0.7529,
        0.3765,    0.2510,    0.2510,
        0.8784,    0.2510,    0.2510,
        0.3765,    0.7529,    0.2510,
        0.8784,    0.7529,    0.2510,
        0.3765,    0.2510,    0.7529,
        0.8784,    0.2510,    0.7529,
        0.3765,    0.7529,    0.7529,
        0.8784,    0.7529,    0.7529,
            0,    0.1255,         0,
        0.5020,    0.1255,         0,
            0,    0.6275,         0,
        0.5020,    0.6275,         0,
            0,    0.1255,    0.5020,
        0.5020,    0.1255,    0.5020,
            0,    0.6275,    0.5020,
        0.5020,    0.6275,    0.5020,
        0.2510,    0.1255,         0,
        0.7529,    0.1255,         0,
        0.2510,    0.6275,         0,
        0.7529,    0.6275,         0,
        0.2510,    0.1255,    0.5020,
        0.7529,    0.1255,    0.5020,
        0.2510,    0.6275,    0.5020,
        0.7529,    0.6275,    0.5020,
            0,    0.3765,         0,
        0.5020,    0.3765,         0,
            0,    0.8784,         0,
        0.5020,    0.8784,         0,
            0,    0.3765,    0.5020,
        0.5020,    0.3765,    0.5020,
            0,    0.8784,    0.5020,
        0.5020,    0.8784,    0.5020,
        0.2510,    0.3765,         0,
        0.7529,    0.3765,         0,
        0.2510,    0.8784,         0,
        0.7529,    0.8784,         0,
        0.2510,    0.3765,    0.5020,
        0.7529,    0.3765,    0.5020,
        0.2510,    0.8784,    0.5020,
        0.7529,    0.8784,    0.5020,
            0,    0.1255,    0.2510,
        0.5020,    0.1255,    0.2510,
            0,    0.6275,    0.2510,
        0.5020,    0.6275,    0.2510,
            0,    0.1255,    0.7529,
        0.5020,    0.1255,    0.7529,
            0,    0.6275,    0.7529,
        0.5020,    0.6275,    0.7529,
        0.2510,    0.1255,    0.2510,
        0.7529,    0.1255,    0.2510,
        0.2510,    0.6275,    0.2510,
        0.7529,    0.6275,    0.2510,
        0.2510,    0.1255,    0.7529,
        0.7529,    0.1255,    0.7529,
        0.2510,    0.6275,    0.7529,
        0.7529,    0.6275,    0.7529,
            0,    0.3765,    0.2510,
        0.5020,    0.3765,    0.2510,
            0,    0.8784,    0.2510,
        0.5020,    0.8784,    0.2510,
            0,    0.3765,    0.7529,
        0.5020,    0.3765,    0.7529,
            0,    0.8784,    0.7529,
        0.5020,    0.8784,    0.7529,
        0.2510,    0.3765,    0.2510,
        0.7529,    0.3765,    0.2510,
        0.2510,    0.8784,    0.2510,
        0.7529,    0.8784,    0.2510,
        0.2510,    0.3765,    0.7529,
        0.7529,    0.3765,    0.7529,
        0.2510,    0.8784,    0.7529,
        0.7529,    0.8784,    0.7529,
        0.1255,    0.1255,         0,
        0.6275,    0.1255,         0,
        0.1255,    0.6275,         0,
        0.6275,    0.6275,         0,
        0.1255,    0.1255,    0.5020,
        0.6275,    0.1255,    0.5020,
        0.1255,    0.6275,    0.5020,
        0.6275,    0.6275,    0.5020,
        0.3765,    0.1255,         0,
        0.8784,    0.1255,         0,
        0.3765,    0.6275,         0,
        0.8784,    0.6275,         0,
        0.3765,    0.1255,    0.5020,
        0.8784,    0.1255,    0.5020,
        0.3765,    0.6275,    0.5020,
        0.8784,    0.6275,    0.5020,
        0.1255,    0.3765,         0,
        0.6275,    0.3765,         0,
        0.1255,    0.8784,         0,
        0.6275,    0.8784,         0,
        0.1255,    0.3765,    0.5020,
        0.6275,    0.3765,    0.5020,
        0.1255,    0.8784,    0.5020,
        0.6275,    0.8784,    0.5020,
        0.3765,    0.3765,         0,
        0.8784,    0.3765,         0,
        0.3765,    0.8784,         0,
        0.8784,    0.8784,         0,
        0.3765,    0.3765,    0.5020,
        0.8784,    0.3765,    0.5020,
        0.3765,    0.8784,    0.5020,
        0.8784,    0.8784,    0.5020,
        0.1255,    0.1255,    0.2510,
        0.6275,    0.1255,    0.2510,
        0.1255,    0.6275,    0.2510,
        0.6275,    0.6275,    0.2510,
        0.1255,    0.1255,    0.7529,
        0.6275,    0.1255,    0.7529,
        0.1255,    0.6275,    0.7529,
        0.6275,    0.6275,    0.7529,
        0.3765,    0.1255,    0.2510,
        0.8784,    0.1255,    0.2510,
        0.3765,    0.6275,    0.2510,
        0.8784,    0.6275,    0.2510,
        0.3765,    0.1255,    0.7529,
        0.8784,    0.1255,    0.7529,
        0.3765,    0.6275,    0.7529,
        0.8784,    0.6275,    0.7529,
        0.1255,    0.3765,    0.2510,
        0.6275,    0.3765,    0.2510,
        0.1255,    0.8784,    0.2510,
        0.6275,    0.8784,    0.2510,
        0.1255,    0.3765,    0.7529,
        0.6275,    0.3765,    0.7529,
        0.1255,    0.8784,    0.7529,
        0.6275,    0.8784,    0.7529,
        0.3765,    0.3765,    0.2510,
        0.8784,    0.3765,    0.2510,
        0.3765,    0.8784,    0.2510,
        0.8784,    0.8784,    0.2510,
        0.3765,    0.3765,    0.7529,
        0.8784,    0.3765,    0.7529,
        0.3765,    0.8784,    0.7529,
        0.8784,    0.8784,    0.7529,
    ]).reshape(-1, 3)

    sh = RGB2SH(base)
    sampled = sh[np.arange(N) % len(sh),:]

    return torch.from_numpy(sampled)

def render_set(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, render_mode="regular", view_idx=0):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    # Measurement
    psnr_list, ssim_list, lpips_list = [], [], []
    ms_ssim_list, alex_lpips_list = [], []

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    renderings = []
    color_embs = get_chaerin_colors(deform.deform.node_num).float().cuda()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpt_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        if deform.name == 'mlp':
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        elif deform.name == 'node':
            time_input = deform.deform.expand_time(fid)
        d_values = deform.step(xyz.detach(), time_input, feature=gaussians.feature, motion_mask=gaussians.motion_mask)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color'] # last two are None here
        render_gs = gaussians
        if render_mode == "lbs":
            lbs_gs = GaussianModel.build_from(gaussians, sh_degree=gaussians.max_sh_degree)
            # lbs_gs._features_dc = get_contrastive_colors(len(lbs_gs._features_dc), dtype=)
            nn_weight, nn_idx = d_values["nn_weight"], d_values["nn_idx"]
            N, K = nn_weight.shape
            num_nodes = deform.deform.node_num
            dense_weight = torch.zeros(N, num_nodes,
                                    device=nn_weight.device,
                                    dtype=nn_weight.dtype)

            dense_weight.scatter_(1, nn_idx, nn_weight)
            mixed_colors = dense_weight @ color_embs
            lbs_gs._features_dc = mixed_colors.unsqueeze(-2)

            render_gs = lbs_gs

        results = render(view, render_gs, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        alpha = results["alpha"]
        rendering = torch.clamp(torch.cat([results["render"], alpha]), 0.0, 1.0)

        if render_mode == "nodes":
            # compute node deltas
            node_gs = deform.deform.as_gaussians_visualization #gaussians + deform.deform.as_gaussians_visualization
            time_input = fid.unsqueeze(0).expand(node_gs.get_xyz.shape[0], -1)
            d_values_node = deform.deform.query_network(x=node_gs.get_xyz.detach(), t=time_input)
            d_xyz_node, d_opacity_node, d_color_node = d_values_node['d_xyz'] * node_gs.motion_mask, None, None
            d_rotation_node, d_scaling_node = torch.zeros_like(node_gs._rotation, device="cuda"), torch.zeros_like(node_gs._scaling, device="cuda")
            results_node = render(view, node_gs, pipeline, background, d_xyz_node, d_rotation_node, d_scaling_node, d_opacity=d_opacity_node, d_color=d_color_node, d_rot_as_res=deform.d_rot_as_res)
            
            rgb_old, alpha_old = rendering[:3, ...], rendering[3:, ...]
            rgb_new, alpha_new = results_node["render"],   results_node["alpha"]
            out_rgb   = rgb_new * alpha_new + rgb_old * (1 - alpha_new)
            out_alpha = alpha_new + alpha_old * (1 - alpha_new)
            rendering = torch.cat((out_rgb, out_alpha), dim=0).clamp(0.0, 1.0)
            # print(d_values['nn_weight'].shape)

            # # append to model gs
            # render_gs = render_gs + node_gs
            # d_xyz = torch.cat([d_xyz, d_xyz_node], dim=0)
            # d_rotation = torch.cat([d_rotation, d_rotation_node], dim=0)
            # d_scaling = torch.cat([d_scaling, d_scaling_node], dim=0)
            # d_opacity = d_color = None

        # Measurement
        image = rendering[:3]
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        psnr_list.append(psnr(image[None], gt_image[None]).mean())
        ssim_list.append(ssim_func(image[None], gt_image[None], data_range=1.).mean())
        lpips_list.append(lpips(image[None], gt_image[None]).mean())
        ms_ssim_list.append(ms_ssim(image[None], gt_image[None], data_range=1.).mean())
        alex_lpips_list.append(alex_lpips(image[None], gt_image[None]).mean())

        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:4, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, f"video_{render_mode}.mp4"), renderings, fps=30, quality=8)

    # Measurement
    psnr_test = torch.stack(psnr_list).mean()
    ssim_test = torch.stack(ssim_list).mean()
    lpips_test = torch.stack(lpips_list).mean()
    ms_ssim_test = torch.stack(ms_ssim_list).mean()
    alex_lpips_test = torch.stack(alex_lpips_list).mean()
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {} MS SSIM{} ALEX_LPIPS {}".format(iteration, name, psnr_test, ssim_test, lpips_test, ms_ssim_test, alex_lpips_test))


def interpolate_time(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, render_mode="regular", view_idx=0):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    view = views[view_idx]
    renderings = []
    color_embs = get_chaerin_colors(deform.deform.node_num).float().cuda()
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        if deform.name == 'deform':
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        elif deform.name == 'node':
            time_input = deform.deform.expand_time(fid)
        d_values = deform.step(xyz.detach(), time_input, feature=gaussians.feature, motion_mask=gaussians.motion_mask)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        render_gs = gaussians
        if render_mode == "lbs":
            lbs_gs = GaussianModel.build_from(gaussians, sh_degree=gaussians.max_sh_degree)
            # lbs_gs._features_dc = get_contrastive_colors(len(lbs_gs._features_dc), dtype=)
            nn_weight, nn_idx = d_values["nn_weight"], d_values["nn_idx"]
            N, K = nn_weight.shape
            num_nodes = deform.deform.node_num
            dense_weight = torch.zeros(N, num_nodes,
                                    device=nn_weight.device,
                                    dtype=nn_weight.dtype)

            dense_weight.scatter_(1, nn_idx, nn_weight)
            mixed_colors = dense_weight @ color_embs
            lbs_gs._features_dc = mixed_colors.unsqueeze(-2)

            render_gs = lbs_gs
            mask = render_gs.prune_cov()
            d_xyz, d_rotation, d_scaling = d_xyz[~mask], d_rotation[~mask], d_scaling[~mask]
        results = render(view, render_gs, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        rendering = results["render"]
        if render_mode == "nodes":
            rendering = torch.clamp(torch.cat([results["render"], results["alpha"]]), 0.0, 1.0)

            # compute node deltas
            node_gs = deform.deform.as_gaussians_visualization #gaussians + deform.deform.as_gaussians_visualization
            time_input = fid.unsqueeze(0).expand(node_gs.get_xyz.shape[0], -1)
            d_values_node = deform.deform.query_network(x=node_gs.get_xyz.detach(), t=time_input)
            d_xyz_node, d_opacity_node, d_color_node = d_values_node['d_xyz'] * node_gs.motion_mask, None, None
            d_rotation_node, d_scaling_node = torch.zeros_like(node_gs._rotation, device="cuda"), torch.zeros_like(node_gs._scaling, device="cuda")
            results_node = render(view, node_gs, pipeline, background, d_xyz_node, d_rotation_node, d_scaling_node, d_opacity=d_opacity_node, d_color=d_color_node, d_rot_as_res=deform.d_rot_as_res)
            
            rgb_old, alpha_old = rendering[:3, ...], rendering[3:, ...]
            rgb_new, alpha_new = results_node["render"],   results_node["alpha"]
            out_rgb   = rgb_new * alpha_new + rgb_old * (1 - alpha_new)
            out_alpha = alpha_new + alpha_old * (1 - alpha_new)
            rendering = torch.cat((out_rgb, out_alpha), dim=0).clamp(0.0, 1.0)[:3,...]

        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, f"video_{render_mode}.mp4"), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, render_mode="regular", view_idx=0):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    # print(views[0].T)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 1.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]], 0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    color_embs = get_chaerin_colors(deform.deform.node_num).float().cuda()
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        if deform.name == 'mlp':
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        elif deform.name == 'node':
            time_input = deform.deform.expand_time(fid)

        d_values = deform.step(xyz.detach(), time_input, feature=gaussians.feature, motion_mask=gaussians.motion_mask)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        render_gs = gaussians
        if render_mode == "lbs":
            render_gs = GaussianModel.build_from(gaussians, sh_degree=gaussians.max_sh_degree)
            lbs_gs = GaussianModel.build_from(render_gs, sh_degree=render_gs.max_sh_degree)
            # lbs_gs._features_dc = get_contrastive_colors(len(lbs_gs._features_dc), dtype=)
            nn_weight, nn_idx = d_values["nn_weight"], d_values["nn_idx"]
            N, K = nn_weight.shape
            idx = nn_idx[torch.arange(N), nn_weight.argmax(dim=1)]
            num_nodes = deform.deform.node_num
            dense_weight = torch.zeros(N, num_nodes,
                                    device=nn_weight.device,
                                    dtype=nn_weight.dtype)
            dense_weight[torch.arange(N), idx] = 1.0

            # dense_weight.scatter_(1, nn_idx, nn_weight)
            mixed_colors = dense_weight @ color_embs
            lbs_gs._features_dc = mixed_colors.unsqueeze(-2)

            render_gs = lbs_gs
            mask = render_gs.prune_cov()
            d_xyz, d_rotation, d_scaling = d_xyz[~mask], d_rotation[~mask], d_scaling[~mask]
        results = render(view, render_gs, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        rendering = torch.clamp(results["render"], 0.0, 1.0)
        if render_mode == "nodes":
            rendering = torch.clamp(torch.cat([results["render"], results["alpha"]]), 0.0, 1.0)

            # compute node deltas
            node_gs = deform.deform.as_gaussians_visualization #gaussians + deform.deform.as_gaussians_visualization
            time_input = fid.unsqueeze(0).expand(node_gs.get_xyz.shape[0], -1)
            d_values_node = deform.deform.query_network(x=node_gs.get_xyz.detach(), t=time_input)
            d_xyz_node, d_opacity_node, d_color_node = d_values_node['d_xyz'] * node_gs.motion_mask, None, None
            d_rotation_node, d_scaling_node = torch.zeros_like(node_gs._rotation, device="cuda"), torch.zeros_like(node_gs._scaling, device="cuda")
            results_node = render(view, node_gs, pipeline, background, d_xyz_node, d_rotation_node, d_scaling_node, d_opacity=d_opacity_node, d_color=d_color_node, d_rot_as_res=deform.d_rot_as_res)
            
            rgb_old, alpha_old = rendering[:3, ...], rendering[3:, ...]
            rgb_new, alpha_new = results_node["render"],   results_node["alpha"]
            out_rgb   = rgb_new * alpha_new + rgb_old * (1 - alpha_new)
            out_alpha = alpha_new + alpha_old * (1 - alpha_new)
            rendering = torch.cat((out_rgb, out_alpha), dim=0).clamp(0.0, 1.0)[:3,...]

        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, f"video_{render_mode}.mp4"), renderings, fps=30, quality=8)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, mode: str, load2device_on_the_fly=False, render_mode="regular", view_idx=0):
    with torch.no_grad():
        
        deform = DeformModel(K=dataset.K, deform_type=dataset.deform_type, is_blender=dataset.is_blender, skinning=dataset.skinning, hyper_dim=dataset.hyper_dim, node_num=dataset.node_num, pred_opacity=dataset.pred_opacity, pred_color=dataset.pred_color, use_hash=dataset.use_hash, hash_time=dataset.hash_time, d_rot_as_res=dataset.d_rot_as_res, local_frame=dataset.local_frame, progressive_brand_time=dataset.progressive_brand_time, max_d_scale=dataset.max_d_scale)
        deform.load_weights(dataset.model_path, iteration=iteration)

        gs_fea_dim = deform.deform.node_num if dataset.skinning and deform.name == 'node' else dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=dataset.gs_with_motion_mask)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, load2device_on_the_fly, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, deform, render_mode=render_mode, view_idx=view_idx)

        if not skip_test:
            render_func(dataset.model_path, load2device_on_the_fly, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, deform, render_mode=render_mode, view_idx=view_idx)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 80_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    # parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')
    parser.add_argument("--render_mode", type=str, default="regular")
    parser.add_argument("--view_idx", type=int, default=0)

    args = get_combined_args(parser)
    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, load2device_on_the_fly=args.load2gpu_on_the_fly, render_mode=args.render_mode, view_idx=args.view_idx)
