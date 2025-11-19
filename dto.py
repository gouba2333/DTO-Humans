import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import cv2
import smplx
import matplotlib
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.render import render_meshes, render_side_views
from utils.color import demo_color as color
from utils.evaluation import calculate_iou

from depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm
import argparse

# Replace neural_renderer with PyTorch3D for depth rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
)

def filter_x(depth_map, transZ_map, mask):
    # ===== 1. Extract original pixels under the mask =====
    y_idx, x_idx = np.where(mask)
    depth_vals_orig = depth_map[mask]
    transZ_vals_orig = transZ_map[mask]

    # ===== 2. Remove Y-axis bias: DepthAnything v2 output has Y-axis bias, only compute X-axis correlation =====
    depth_xnorm = np.zeros_like(depth_vals_orig, dtype=np.float32)
    transZ_xnorm = np.zeros_like(transZ_vals_orig, dtype=np.float32)

    unique_y = np.unique(y_idx)
    for y in unique_y:
        row_mask = (y_idx == y)
        row_depth_vals = depth_vals_orig[row_mask]
        row_transZ_vals = transZ_vals_orig[row_mask]
        depth_xnorm[row_mask] = row_depth_vals - np.mean(row_depth_vals)
        transZ_xnorm[row_mask] = row_transZ_vals - np.mean(row_transZ_vals)

    slope, intercept = np.polyfit(depth_xnorm, transZ_xnorm, 1)
    corr = np.corrcoef(depth_xnorm, transZ_xnorm)[0, 1]
    if np.isnan(corr):
        corr = 0.0

    # ===== 3. Identify and remove outliers =====
    threshold = 0.1
    predicted_transZ_xnorm = depth_xnorm * slope + intercept
    residuals = np.abs(predicted_transZ_xnorm - transZ_xnorm)
    is_inlier = residuals <= threshold
    
    inlier_y_coords = y_idx[is_inlier]
    inlier_x_coords = x_idx[is_inlier]
    
    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[inlier_y_coords, inlier_x_coords] = True
   
    return slope, corr, new_mask

class Optimizer:
    def __init__(self, device, args):
        self.device = device
        self.smpl_model_type = 'smpl' if not args.use_smpla else 'smpla'
        self.use_age_prior = args.use_age_prior
        self.use_gender_prior = args.use_gender_prior
        self.use_X_cond = args.use_X_cond
        self.step = 0
        self.vis_interval = args.vis_step
        print("Loading Depth Anything v2...")
        self.load_depth_anything_v2()
        self.load_smpl_model()
    
    def load_smpl_model(self):
        from utils.smpl_wrapper import SMPL # Use 4D-humans SMPL model
        smpl_cfg = {
            "model_path": "data/models/smpl",
            "gender": "neutral",
            "model_type": self.smpl_model_type,
            "joint_regressor_extra": "data/models/smpl/SMPL_to_J19.pkl"
        }
        self.smpl_model = SMPL(**smpl_cfg).to(self.device)

    def load_depth_anything_v2(self):
        model = 'vitl'
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        checkpoint=f'ckpt/depth_anything_v2_{model}.pth'
        self.depth_anything: nn.Module = DepthAnythingV2(**model_configs[model])
        state_dict = torch.load(checkpoint, map_location='cpu')
        self.depth_anything.load_state_dict(state_dict)
        self.depth_anything = self.depth_anything.to(self.device).eval()
    
    def get_height(self, betas):
        output = self.smpl_model(
            betas=torch.tensor(betas).unsqueeze(0).to(self.device),
            global_orient=torch.zeros(1, 1, 3).to(self.device),
            body_pose=torch.zeros(1, 23, 3).to(self.device),
            transl=torch.zeros(1, 3).to(self.device),
        )
        verts = output.vertices.detach().cpu().numpy()[0]
        height = np.max(verts[:, 1]) - np.min(verts[:, 1])
        return height

    def calculate_visibility_with_renderer(
        self, seg_list, pred_verts, faces, K, R, t, img_size_H, img_size_W, render_res_max=512
    ):
        device = pred_verts.device
        B, V, _ = pred_verts.shape

        # Downscale for faster rendering; if original resolution is already <= target, no scaling needed
        if max(img_size_H, img_size_W) <= render_res_max:
            scale = 1.0
            render_size_H, render_size_W = img_size_H, img_size_W
        else:
            # Compute scaling ratio
            scale = render_res_max / max(img_size_H, img_size_W)
            render_size_H = int(img_size_H * scale)
            render_size_W = int(img_size_W * scale)
        render_size = (render_size_H, render_size_W)

        # Build intrinsics per person
        if K.dim() == 2:
            K = K.unsqueeze(0)
        if K.shape[0] == 1 and B > 1:
            K = K.expand(B, -1, -1)

        # --- Scale camera intrinsics ---
        K_render = K.clone()
        if scale != 1.0:
            K_render[:, 0, 0] *= scale  # fx
            K_render[:, 1, 1] *= scale  # fy
            K_render[:, 0, 2] *= scale  # px
            K_render[:, 1, 2] *= scale  # py

        fx = K_render[:, 0, 0]
        fy = K_render[:, 1, 1]
        px = K_render[:, 0, 2]
        py = K_render[:, 1, 2]

        cameras = PerspectiveCameras(
            focal_length=torch.stack([fx, fy], dim=-1),
            principal_point=torch.stack([px, py], dim=-1),
            R=torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1),
            T=torch.zeros(B, 3, device=device),
            image_size=(render_size,) * B,
            in_ndc=False,
            device=device,
        )

        # Meshes per person
        faces_t = torch.as_tensor(faces.astype(np.int64), device=device)
        meshes = Meshes(verts=[pred_verts[i] for i in range(B)], faces=[faces_t for _ in range(B)])

        raster_settings = RasterizationSettings(
            image_size=render_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            bin_size=0,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(meshes)
        # zbuf: (B, render_H, render_W, 1); pix_to_face: (B, render_H, render_W, 1)
        zbuf = fragments.zbuf[..., 0]  # (B, render_H, render_W)
        pix_to_face = fragments.pix_to_face[..., 0]

        # Flip vertically to correct PyTorch3D's upside-down output
        zbuf = torch.flip(zbuf, dims=[1, 2])
        pix_to_face = torch.flip(pix_to_face, dims=[1, 2])

        # Set no-hit pixels to +inf
        zbuf = torch.where(pix_to_face < 0, torch.full_like(zbuf, float('inf')), zbuf)

        # Prepare segmentation masks (B, img_size_H, img_size_W)
        seg_tensor_full_res = torch.zeros((B, img_size_H, img_size_W), dtype=torch.bool, device=device)
        for i, seg in enumerate(seg_list):
            if seg is not None:
                seg_tensor_full_res[i] = torch.from_numpy(seg).to(device)

        # Resize full-resolution mask to rendering resolution
        seg_tensor = F.interpolate(
            seg_tensor_full_res.float().unsqueeze(1),
            size=render_size,
            mode='nearest'
        ).squeeze(1).bool()

        # Apply masks: keep depth where seg is true, else +inf
        zbuf_masked = torch.where(seg_tensor, zbuf, torch.full_like(zbuf, float('inf')))

        # Compose nearest-person map across batch
        final_depth_map, min_indices = torch.min(zbuf_masked, dim=0)  # (render_H, render_W)

        # Background: where depth is infinite
        is_background = torch.isinf(final_depth_map)
        final_id_map = min_indices + 1
        final_id_map[is_background] = 0
        final_depth_map = torch.where(is_background, torch.zeros_like(final_depth_map), final_depth_map)

        # Per-vertex visibility via sampling
        orig_fx = K[:, 0, 0]
        orig_fy = K[:, 1, 1]
        orig_px = K[:, 0, 2]
        orig_py = K[:, 1, 2]
        X = pred_verts[..., 0]
        Y = pred_verts[..., 1]
        Z = pred_verts[..., 2].clamp(min=1e-6)
        u = orig_fx.view(B, 1) * (X / Z) + orig_px.view(B, 1)
        v = orig_fy.view(B, 1) * (Y / Z) + orig_py.view(B, 1)

        # Normalize to [-1,1] for grid_sample using original image size
        u_norm = (u / (img_size_W - 1)) * 2 - 1
        v_norm = (v / (img_size_H - 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2)  # (B, V, 1, 2)

        sampled_final_depth = F.grid_sample(
            final_depth_map.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1),
            grid,
            align_corners=True,
            mode='nearest',
            padding_mode='zeros',
        ).squeeze(-1).squeeze(-2)  # (B, V)

        tolerance = 1e-2
        is_visible_mask = (Z <= sampled_final_depth + tolerance) & (sampled_final_depth > 0)
        visible_vert_counts = torch.sum(is_visible_mask, dim=1).detach().cpu().numpy()

        # --- Upsample final output to match original image resolution ---
        if scale != 1.0:
            final_id_map_full_res = F.interpolate(
                final_id_map.float().unsqueeze(0).unsqueeze(0),
                size=(img_size_H, img_size_W),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()

            final_depth_map_full_res = F.interpolate(
                final_depth_map.unsqueeze(0).unsqueeze(0),
                size=(img_size_H, img_size_W),
                mode='nearest'
            ).squeeze(0).squeeze(0)
        else:
            final_id_map_full_res = final_id_map
            final_depth_map_full_res = final_depth_map

        tz_map = torch.stack([final_id_map_full_res, final_depth_map_full_res], dim=-1).detach().cpu().numpy()

        return tz_map, visible_vert_counts

    def optimize(self, raw_image, img_path, annotation, output_dir='output', vis=False, skip_optimize=False):
        # reoptimize = False
        new_annot = defaultdict(list)
        H, W = raw_image.shape[:2]
        is_optimize = False

        if annotation['shape'].shape[-1] == 10 and self.smpl_model_type == 'smpla': # smpla has 11 dim
            annotation['shape'] = np.concatenate([annotation['shape'], np.zeros((len(annotation['shape']), 1), dtype=np.float32)], axis=-1)
        
        # Skip optimization
        if len(annotation['shape']) <= 1 or skip_optimize:
            print(f"skip optimization")
            for i in range(len(annotation['shape'])):
                new_annot['pose_cam'].append(annotation['pose_cam'][i].copy())
                new_annot['shape'].append(annotation['shape'][i].copy())
                new_annot['cam_int'].append(annotation['cam_int'][i].copy())
                new_annot['trans_cam'].append(annotation['trans_cam'][i].copy())
                new_annot['is_released'].append(annotation['is_released'][i])
        
            new_annot = {k: np.stack(v) for k, v in new_annot.items()}
            new_annot['is_optimize'] = is_optimize
            self.step += 1
            return new_annot, is_optimize, 0
        
        dist = self.depth_anything.infer_image(raw_image)
        dist[dist==0] = min(dist[dist>0])  # Replace zero values with minimum non-zero value
        depth = 1/(dist)  # Convert to depth map

        img_basename = '_'.join(img_path.split('/')[-3:])

        # Human prior
        age_split = [3, 8, 15]
        sigma = [0.126, 0.12, 0.156, 0.0707, 0.0759, 0.1]
        mu = [0.801, 1.122, 1.477, 1.647, 1.784, 1.715]

        corr_thresh = 0.5
        gender_thresh = 0.33
        def get_alpha_beta(id_list, indicator_list, gender_list, height_list, tz_list, depth_list, depth_std_list, X_cond=[], corrs=[], max_iter=10):
            # Convert all input lists to float64 numpy arrays for precision
            id_list = np.array(id_list, dtype=np.int32)
            indicator_list = np.array(indicator_list, dtype=np.int32)
            sigma_list = np.array([sigma[indicator] for indicator in indicator_list], dtype=np.float64)
            mu_list = np.array([mu[indicator] for indicator in indicator_list], dtype=np.float64)
            height_list = np.array(height_list, dtype=np.float64)
            tz_list = np.array(tz_list, dtype=np.float64)
            depth_list = np.array(depth_list, dtype=np.float64)
            X_cond = np.array(X_cond, dtype=np.float64) if len(X_cond) > 0 else []
            corrs = np.array(corrs, dtype=np.float64) if len(corrs) > 0 else []

            id_skip_local = []

            if not self.use_gender_prior:
                for i in range(len(indicator_list)):
                    if indicator_list[i] in [3, 4, 5]:
                        indicator_list[i] = 5
                        sigma_list[i] = 0.073
                        mu_list[i] = height_list[i]
            else:
                for i in range(len(indicator_list)):
                    if indicator_list[i] in [3, 4, 5]:
                        mu_list[i] = (height_list[i] + mu_list[i])/2

            indicator_arr = np.array(indicator_list)
            
            for iter in range(max_iter):
                # Solve optimization problem
                A = B = C = D = E = np.float64(0)
                num_opt = 0
                for i in range(len(id_list)):
                    if id_list[i] in id_skip_local: continue
                    num_opt += 1
                    inv_sigma2 = np.float64(1) / (sigma_list[i] ** 2)
                    htz = height_list[i] / tz_list[i]
                    htz2 = htz ** 2
                    d = depth_list[i]
                    m = mu_list[i]
                    A += inv_sigma2 * htz2 * d ** 2
                    B += inv_sigma2 * htz2 * d
                    C += inv_sigma2 * htz2
                    D += inv_sigma2 * htz * d * m
                    E += inv_sigma2 * htz * m
                denom = A * C - B ** 2
                alpha = (C * D - B * E) / denom
                beta = (A * E - B * D) / denom

                Xmin = np.float64(20) # default
                X_cond_sel = X_cond[corrs > corr_thresh]
                corrs_sel = corrs[corrs > corr_thresh]
                print(self.use_X_cond)
                if len(X_cond_sel) > 0 and self.use_X_cond:
                    weights = corrs_sel / np.sum(corrs_sel)
                    Xmin = np.sum(X_cond_sel * weights) # Compute Xmin
                
                depth_used = [depth_list[i] for i in range(len(depth_list)) if id_list[i] not in id_skip_local]
                depth_std_used = [depth_std_list[i] for i in range(len(depth_std_list)) if id_list[i] not in id_skip_local]
                
                # Check for pseudo-plane
                is_line = np.std(depth_used) < np.sum(depth_std_used)/len(depth_std_used)
                
                if alpha < Xmin or is_line: # If alpha is too small or pseudo-plane, use X_cond as alpha
                    # print(np.std(depth_used), np.sum(depth_std_used), len(depth_std_used), is_line)
                    print(f"step {self.step}, {img_basename}, force X from {alpha} to {Xmin}")
                    alpha = Xmin
                    beta = (E - B * alpha) / C
                elif num_opt == 2: # Only two people with height prior and no internal size constraint, not considered optimization
                    return alpha, beta, 0, indicator_arr, 0.0, id_skip_local

                # After optimization, skip the most abnormal height and recalculate for others
                nh_list = []
                worst_id = -1
                worst_err = -1
                for i in range(len(id_list)):
                    if id_list[i] in id_skip_local:
                        nh_list.append(height_list[i])
                        continue
                    new_height = height_list[i] * (alpha * depth_list[i] + beta) / tz_list[i]
                    nh_list.append(new_height)
                    if new_height > 2.1:
                        err = new_height - 2.1
                        if err > worst_err:
                            worst_err = err
                            worst_id = id_list[i]

                if worst_id != -1:
                    id_skip_local.append(worst_id.item())
                    if num_opt <= 2:
                        id_skip_local.extend([id for id in id_list if id not in id_skip_local])
                        return 0, np.max(tz_list), 0, indicator_arr, 0.0, id_skip_local
                    continue

                # Exit condition: no abnormal height
                break

            # Compute mean standardized height deviation
            height_deviations = []
            for i in range(len(id_list)):
                if id_list[i] not in id_skip_local:
                    height_deviation = np.abs(nh_list[i] - mu_list[i]) / sigma_list[i]
                    height_deviations.append(height_deviation)
            
            mean_height_deviation = np.mean(height_deviations) if len(height_deviations) > 0 else 0.0

            return alpha, beta, num_opt, indicator_arr, mean_height_deviation, id_skip_local

        verts_list = []
        height_list = []
        faces_list = []
        trans_list = []
        kp2ds_list = []
        indicator_list = []
        box_list = annotation['bbox']
        seg_list = annotation['seg']

        for id in range(len(annotation['cam_int'])):
            K = annotation['cam_int'][id] # (3, 3)
            betas = annotation['shape'][id] # (, 10)
            pose = annotation['pose_cam'][id] # (24, 3)
            trans = annotation['trans_cam'][id] # (, 3)
            kp2ds_list.append(annotation['kpts_2d'][id])
            trans_list.append(trans.copy())
            output = self.smpl_model(
                betas=torch.tensor(betas).unsqueeze(0).to(self.device),
                global_orient=torch.tensor(pose[0:1]).unsqueeze(0).to(self.device),
                body_pose=torch.tensor(pose[1:]).unsqueeze(0).to(self.device),
                transl=torch.tensor(trans).unsqueeze(0).to(self.device),
            )
            focal = np.asarray([K[0,0],K[1,1]])
            princpt = np.asarray([K[0,-1],K[1,-1]])
            verts = output.vertices.detach().cpu().numpy()[0]
            verts_list.append(verts)
            faces_list.append(self.smpl_model.faces)
            height_list.append(self.get_height(betas))
            gender = annotation['gender'][id]
            age = annotation['age'][id]
            if self.use_age_prior:
                if age is None:
                    indicator_list.append(None)
                elif age >= age_split[2]: # adult
                    if gender < gender_thresh:
                        indicator_list.append(3)
                    elif gender >= 1 - gender_thresh:
                        indicator_list.append(4)
                    else:
                        indicator_list.append(5) # neutral adult
                elif age >= age_split[1]: # teen
                    indicator_list.append(2)
                elif age >= age_split[0]: # kid
                    indicator_list.append(1)
                else: # baby
                    indicator_list.append(0)
            else:
                if gender < gender_thresh:
                    indicator_list.append(3)
                elif gender >= 1 - gender_thresh:
                    indicator_list.append(4)
                else:
                    indicator_list.append(5) # neutral adult

        # By rendering, the translation z and human id corresponding to each pixel point can be obtained
        # in order to obtain the relationship between each person's translation z and depth
        tz_map, visible_vert_counts = self.calculate_visibility_with_renderer(seg_list, torch.from_numpy(np.stack(verts_list)).to(self.device), self.smpl_model.faces, 
                                                                              torch.from_numpy(K).to(self.device).unsqueeze(0), 
                                                                              torch.eye(3, device=self.device).unsqueeze(0), 
                                                                              torch.zeros(3, device=self.device).unsqueeze(0), H, W)
        
        id_rm = []
        id_skip = []
        id_near = []
        id_far = []
        deviation = []

        depth_near_list = []
        depth_std_near_list = []
        depth_far_list = []
        depth_std_far_list = []

        tz_near_list = []
        tz_far_list = []

        use_down = True 
        # Use mean of points near the ground. Improves results for clear ground scenes (e.g., mupots), 
        # but reduce performance for RH (PCDR 74.68 -> 74.17). Set True for final DTO-Humans generation.
        depth_near_down_list = []
        tz_near_down_list = []

        height_near_list = []
        height_far_list = []
        indicator_near_list = []
        indicator_far_list = []
        gender_near_list = []
        gender_far_list = []

        slopes_near = []
        slopes_far = []
        corrs_near = []
        corrs_far = []
        for i in range(len(verts_list)):
            if (i + 1) in id_skip:
                continue
            mask_i = (tz_map[:, :, 0] == (i + 1))
            
            if mask_i.sum() < H/20*W/20 or visible_vert_counts[i] < 500: # Person too small or too few visible vertices
                id_rm.append(i + 1)
                id_skip.append(i + 1)
                tz_map[:, :, 0][mask_i] = 0
                continue

            depth_temp = depth[:, :][mask_i]
            Q1 = np.percentile(depth_temp, 25)
            Q3 = np.percentile(depth_temp, 75)
            # Calculate IQR and filter outliers
            IQR = Q3 - Q1
            lower_bound = max(Q1 - 1.5 * IQR, 0)
            upper_bound = Q3 + 1.5 * IQR
            mask_i = mask_i & (depth > lower_bound) & (depth < upper_bound)

            tz_map[:, :, 0][~mask_i & (tz_map[:, :, 0] == (i + 1))] = 0

            if mask_i.sum() < H/20*W/20: # Person too small after filtering
                id_rm.append(i + 1)
                id_skip.append(i + 1)
                tz_map[:, :, 0][mask_i] = 0
                continue

            slope, corr, new_mask = filter_x(depth, tz_map[:, :, 1], mask_i)

            mask_i = new_mask
            tz_map[:, :, 0][~mask_i & (tz_map[:, :, 0] == (i + 1))] = 0

            depth_values = depth[mask_i]
            tz_values = tz_map[:, :, 1][mask_i]

            human_depth = depth_values.mean()
            human_tz = tz_values.mean()

            y_value = np.where(mask_i)[0]
            y_down = np.percentile(y_value, 90)
            down_mask = (y_value >= y_down)
            human_depth_down = depth[mask_i][down_mask].mean()
            human_tz_down = tz_map[:, :, 1][mask_i][down_mask].mean()

            scale = focal[1] * 2 / trans_list[i][-1] / min(H, W)
            # To prevent distant views with low depth accuracy from affecting close range, optimize separately
            depth_thresh = 0.02
            scale_thresh = 1/4
            if human_depth > depth_thresh or scale < scale_thresh:
                print(f"Human {i+1} in {img_basename} is far, depth: {human_depth}")
                id_far.append(i + 1)
                depth_far_list.append(human_depth)
                depth_std_far_list.append(depth_values.std())
                tz_far_list.append(human_tz)
                height_far_list.append(height_list[i])
                indicator_far_list.append(indicator_list[i])
                gender_far_list.append(annotation['gender'][i])
                slopes_far.append(slope) 
                corrs_far.append(corr)
            else:
                id_near.append(i + 1)
                depth_near_list.append(human_depth)
                depth_near_down_list.append(human_depth_down)
                depth_std_near_list.append(depth_values.std())
                tz_near_list.append(human_tz)
                tz_near_down_list.append(human_tz_down)
                height_near_list.append(height_list[i])
                indicator_near_list.append(indicator_list[i])
                gender_near_list.append(annotation['gender'][i])
                slopes_near.append(slope)
                corrs_near.append(corr)

        new_height_list = []
        new_box_list = []
        new_indicator_list = []
        new_focal = focal.copy() 
        # TODO: Change focal length using depth cues. Current depth model not accurate enough; 
        # optimizing focal length via internal body depth is not effective.
        
        # Near scene optimization
        alpha_near, beta_near = 0.0, 0.0
        if len(id_near) > 0:
            if len(id_near) == 1:
                alpha_near = 0
                beta_near = tz_near_list[0]
                id_skip.append(id_near[0])
                print(f"step {self.step}, {img_basename}, only one near person, no near optimization")
            else:
                alpha_near, beta_near, optimize_num, indicator_list_update, mean_height_deviation_near, id_skip_local_near = get_alpha_beta(id_near, indicator_near_list, gender_near_list, height_near_list, tz_near_list, depth_near_list, depth_std_near_list, X_cond=slopes_near, corrs=corrs_near)
                alpha_near_down, beta_near_down, optimize_num_down, indicator_list_update_down, mean_height_deviation_near_down, id_skip_local_near_down = get_alpha_beta(id_near, indicator_near_list, gender_near_list, height_near_list, tz_near_down_list, depth_near_down_list, depth_std_near_list, X_cond=slopes_near, corrs=corrs_near)
                if use_down and optimize_num_down >= optimize_num and mean_height_deviation_near_down < mean_height_deviation_near:
                    # print(f"step {self.step}, {img_basename}, use down y for near optimization")
                    alpha_near = alpha_near_down
                    beta_near = beta_near_down
                    indicator_list_update = indicator_list_update_down
                    mean_height_deviation_near = mean_height_deviation_near_down
                    optimize_num = optimize_num_down
                    id_skip_local_near = id_skip_local_near_down
                    depth_near_list = depth_near_down_list
                    tz_near_list = tz_near_down_list
                id_skip.extend(id_skip_local_near)
                # Skipped persons are counted as 3x deviation
                deviation = deviation + [mean_height_deviation_near] * optimize_num + [3] * len(id_skip_local_near)
                print(f"step {self.step}, {img_basename}, near mean height deviation: {mean_height_deviation_near:.3f}")
                is_optimize = is_optimize or (optimize_num >= 2)
                for i, human_tz, human_depth, indicator in zip(id_near, tz_near_list, depth_near_list, indicator_list_update):
                    if i in id_skip: continue
                    human_shift = (human_depth * alpha_near + beta_near) / human_tz
                    new_height_list.append(height_list[i-1] * human_shift)
                    new_box_list.append(box_list[i-1])
                    new_indicator_list.append(indicator)
                    # Update annotation
                    new_annot['pose_cam'].append(annotation['pose_cam'][i-1].copy())
                    new_shape = annotation['shape'][i-1].copy()
                    # If SMPLA and child (indicator 0/1), adjust last beta; otherwise adjust first beta
                    beta_idx = -1 if (self.smpl_model_type == 'smpla' and (indicator in (0, 1, 2) and human_shift < 1 or new_height_list[-1] < 1.45)) else 0
                    shape_minus_one = new_shape.copy()
                    shape_minus_one[beta_idx] = shape_minus_one[beta_idx] - 1
                    denom_height = self.get_height(shape_minus_one)
                    new_shape[beta_idx] = new_shape[beta_idx] + height_list[i-1] * (human_shift - 1) / (height_list[i-1] - denom_height)
                    new_annot['shape'].append(new_shape)
                    new_cam = annotation['cam_int'][i-1].copy()
                    new_cam[0,0] = new_focal[0]
                    new_cam[1,1] = new_focal[1]
                    new_annot['cam_int'].append(new_cam)
                    new_trans = annotation['trans_cam'][i-1].copy()
                    new_trans = new_trans * human_shift
                    new_trans[2] = new_trans[2] * new_focal[0] / focal[0]
                    # If SMPLA and child (indicator 0/1) and last beta used,
                    # use least squares (mean offset) to fine-tune tx,ty to minimize difference for 2D joints
                    if beta_idx == -1:
                        with torch.no_grad():
                            pose_i = annotation['pose_cam'][i-1]
                            trans_orig = annotation['trans_cam'][i-1]
                            betas_orig = annotation['shape'][i-1]
                            out_orig = self.smpl_model(
                                betas=torch.tensor(betas_orig).unsqueeze(0).to(self.device),
                                global_orient=torch.tensor(pose_i[0:1]).unsqueeze(0).to(self.device),
                                body_pose=torch.tensor(pose_i[1:]).unsqueeze(0).to(self.device),
                                transl=torch.tensor(trans_orig).unsqueeze(0).to(self.device),
                            )
                            joints_orig = out_orig.joints[0, :24].detach().cpu().numpy()

                            out_new = self.smpl_model(
                                betas=torch.tensor(new_shape).unsqueeze(0).to(self.device),
                                global_orient=torch.tensor(pose_i[0:1]).unsqueeze(0).to(self.device),
                                body_pose=torch.tensor(pose_i[1:]).unsqueeze(0).to(self.device),
                                transl=torch.tensor(new_trans).unsqueeze(0).to(self.device),
                            )
                            joints_new = out_new.joints[0, :24].detach().cpu().numpy()

                        delta_xy = (joints_orig[:, :2] * human_shift - joints_new[:, :2]).mean(axis=0)
                        new_trans[0] += float(delta_xy[0])
                        new_trans[1] += float(delta_xy[1])
                    new_annot['trans_cam'].append(new_trans)
                    new_annot['is_released'].append(annotation['is_released'][i-1])

        # Far scene optimization
        alpha_far, beta_far = 0.0, 0.0
        if len(id_far) > 0:
            if len(id_far) == 1:
                alpha_far = 0
                beta_far = tz_far_list[0]
                id_skip.append(id_far[0])
                print(f"step {self.step}, {img_basename}, only one far person, no far optimization")
            else:
                alpha_far, beta_far, optimize_num, indicator_list_update, mean_height_deviation_far, id_skip_local_far = get_alpha_beta(id_far, indicator_far_list, gender_far_list, height_far_list, tz_far_list, depth_far_list, depth_std_far_list, X_cond=slopes_far, corrs=corrs_far)
                id_skip.extend(id_skip_local_far)
                if len(id_near) > 0:
                    beta_far = max(beta_far, np.max(depth_near_list) * alpha_near + beta_near - np.min(depth_far_list) * alpha_far)
                    # Recalculate height with beta_far; skip if height > 2.1
                    for i, id in enumerate(id_far):
                        if id in id_skip_local_far:
                            continue
                        human_tz = tz_far_list[i]
                        human_depth = depth_far_list[i]
                        human_shift = (human_depth * alpha_far + beta_far) / human_tz
                        new_height = height_list[id-1] * human_shift
                        if new_height > 2.1:
                            optimize_num -= 1
                            print(f"step {self.step}, {img_basename}, after near adjust, height {new_height:.3f} for id {id}, skip it")
                            id_skip_local_far.append(id)
                            id_skip.append(id)
                
                is_optimize = is_optimize or (optimize_num >= 2)
                deviation = deviation + [mean_height_deviation_far] * optimize_num + [3] * len(id_skip_local_far)

                for i, human_tz, human_depth, indicator in zip(id_far, tz_far_list, depth_far_list, indicator_list_update):
                    if i in id_skip: continue
                    human_shift = (human_depth * alpha_far + beta_far) / human_tz
                    new_height_list.append(height_list[i-1] * human_shift)
                    new_box_list.append(box_list[i-1])
                    new_indicator_list.append(indicator)
                    # Update annotation
                    new_annot['pose_cam'].append(annotation['pose_cam'][i-1].copy())
                    new_shape = annotation['shape'][i-1].copy()
                    # If SMPLA and child (indicator 0/1), adjust last beta; otherwise adjust first beta
                    beta_idx = -1 if (self.smpl_model_type == 'smpla' and (indicator in (0, 1, 2) and human_shift < 1 or new_height_list[-1] < 1.45)) else 0
                    shape_minus_one = new_shape.copy()
                    shape_minus_one[beta_idx] = shape_minus_one[beta_idx] - 1
                    denom_height = self.get_height(shape_minus_one)
                    new_shape[beta_idx] = new_shape[beta_idx] + height_list[i-1] * (human_shift - 1) / (height_list[i-1] - denom_height)
                    new_annot['shape'].append(new_shape)
                    new_cam = annotation['cam_int'][i-1].copy()
                    new_cam[0,0] = new_focal[0]
                    new_cam[1,1] = new_focal[1]
                    new_annot['cam_int'].append(new_cam)
                    new_trans = annotation['trans_cam'][i-1].copy()
                    new_trans = new_trans * human_shift
                    new_trans[2] = new_trans[2] * new_focal[0] / focal[0]
                    if beta_idx == -1:
                        with torch.no_grad():
                            pose_i = annotation['pose_cam'][i-1]
                            trans_orig = annotation['trans_cam'][i-1]
                            betas_orig = annotation['shape'][i-1]
                            out_orig = self.smpl_model(
                                betas=torch.tensor(betas_orig).unsqueeze(0).to(self.device),
                                global_orient=torch.tensor(pose_i[0:1]).unsqueeze(0).to(self.device),
                                body_pose=torch.tensor(pose_i[1:]).unsqueeze(0).to(self.device),
                                transl=torch.tensor(trans_orig).unsqueeze(0).to(self.device),
                            )
                            joints_orig = out_orig.joints[0, :24].detach().cpu().numpy()

                            out_new = self.smpl_model(
                                betas=torch.tensor(new_shape).unsqueeze(0).to(self.device),
                                global_orient=torch.tensor(pose_i[0:1]).unsqueeze(0).to(self.device),
                                body_pose=torch.tensor(pose_i[1:]).unsqueeze(0).to(self.device),
                                transl=torch.tensor(new_trans).unsqueeze(0).to(self.device),
                            )
                            joints_new = out_new.joints[0, :24].detach().cpu().numpy()

                        delta_xy = (joints_orig[:, :2] * human_shift - joints_new[:, :2]).mean(axis=0)
                        new_trans[0] += float(delta_xy[0])
                        new_trans[1] += float(delta_xy[1])
                    new_annot['trans_cam'].append(new_trans)
                    new_annot['is_released'].append(annotation['is_released'][i-1])
        
        if len(id_skip) > 0:
            for i in id_skip:
                if i in id_rm:
                    continue
                new_height_list.append(height_list[i-1])
                new_box_list.append(box_list[i-1])
                new_indicator_list.append(None)
                # Update annotation
                new_annot['pose_cam'].append(annotation['pose_cam'][i-1].copy())
                new_annot['shape'].append(annotation['shape'][i-1].copy())
                new_cam = annotation['cam_int'][i-1].copy()
                new_cam[0,0] = new_focal[0]
                new_cam[1,1] = new_focal[1]
                new_annot['cam_int'].append(new_cam)
                new_trans = annotation['trans_cam'][i-1].copy()
                new_trans[2] = new_trans[2] * new_focal[0] / focal[0]
                new_annot['trans_cam'].append(new_trans)
                new_annot['is_released'].append(annotation['is_released'][i-1])
        
        new_annot = {k: np.stack(v) for k, v in new_annot.items()}
        mean_height_deviation = np.mean(deviation) if len(deviation) > 0 else 0.0
        new_annot['is_optimize'] = is_optimize
        if vis and is_optimize and self.step % self.vis_interval == 0:
            self.visualize(
                img_path, output_dir,
                raw_image, verts_list, faces_list, indicator_list, box_list, height_list, trans_list,
                new_annot, new_indicator_list, new_box_list, new_height_list, tz_map, 
                focal, new_focal, princpt, dist, 1/depth, K, mean_height_deviation
            )

        self.step += 1
        return new_annot, is_optimize, mean_height_deviation

    def visualize(
            self, img_path, output_dir, 
            raw_image, verts_list, faces_list, indicator_list, box_list, height_list, trans_list,
            new_annot, new_indicator_list, new_box_list, new_height_list, tz_map,
            focal, new_focal, princpt, dist, dist_new, K, mean_height_deviation, resize_size=1024
        ):
        new_trans_list = new_annot['trans_cam']
        new_K = new_annot['cam_int'][0]
        output = self.smpl_model(
            betas=torch.tensor(new_annot['shape']).to(self.device),
            global_orient=torch.tensor(new_annot['pose_cam'][:, 0:1]).to(self.device),
            body_pose=torch.tensor(new_annot['pose_cam'][:, 1:]).to(self.device),
            transl=torch.tensor(new_annot['trans_cam']).to(self.device),
        )
        new_verts_list = output.vertices.detach().cpu().numpy()
        
        # Draw bounding boxes and annotate with height and indicator
        def draw_boxes(image, indicators, boxes, heights):
            for indicator, box, height in zip(indicators, boxes, heights):
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                indicator_dict = {0: "C1", 1: "C2", 2: "C3", 3: "F", 4: "M", 5:"N"}
                if indicator is not None:
                    cv2.putText(image, f'{indicator_dict[indicator]} {height:.2f}', (int(x1), int(y1+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return image
        
        h, w, _ = raw_image.shape
        downsample_factor = max(h, w) / resize_size if max(h, w) > resize_size else 1
        new_h, new_w = int(h / downsample_factor), int(w / downsample_factor)
        raw_image_downsampled = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        focal_downsampled = focal / downsample_factor
        new_focal_downsampled = new_focal / downsample_factor
        princpt_downsampled = princpt / downsample_factor
        # Adjust K matrix for render_side_views
        K_downsampled = K.copy()
        K_downsampled[:2, :] /= downsample_factor
        new_K_downsampled = new_K.copy()
        new_K_downsampled[:2, :] /= downsample_factor

        box_list_downsampled = [(box/downsample_factor).astype(int) if box is not None else None for box in box_list]
        new_box_list_downsampled = [(box/downsample_factor).astype(int) if box is not None else None for box in new_box_list]

        pred_rend_array_init = render_meshes(raw_image_downsampled, verts_list, faces_list,
                        {'focal': focal_downsampled, 'princpt': princpt_downsampled}, color=color)
        # pred_rend_array_init = draw_boxes(pred_rend_array_init, indicator_list, box_list_downsampled, height_list)

        pred_rend_array = render_meshes(raw_image_downsampled, new_verts_list, faces_list,
                        {'focal': new_focal_downsampled, 'princpt': princpt_downsampled}, color=color)
        # pred_rend_array = draw_boxes(pred_rend_array, new_indicator_list, new_box_list_downsampled, new_height_list)

        _, _, pred_rend_array_bev_init = render_side_views(raw_image_downsampled, color, verts_list, faces_list, trans_list, K_downsampled, render_bev=True)
        _, _, pred_rend_array_bev = render_side_views(raw_image_downsampled, color, new_verts_list, faces_list, new_trans_list, new_K_downsampled, render_bev=True)

        cmap = matplotlib.colormaps.get_cmap('Spectral_r')

        dist_full = np.asarray(dist)
        dist_new_full = np.asarray(dist_new)

        # Colorize full-res dist map
        dmin, dmax = dist_full.min(), dist_full.max()
        if dmax == dmin:
            dist_color_full = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            dist_norm = ((dist_full - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
            dist_color_full = (cmap(dist_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Human mask at full resolution from tz_map
        human_mask_full = tz_map[:, :, 0] > 0
        dist_color_human_full = dist_color_full.copy()

        # Compute human-only colored map using dist_new (full-res) for normalization
        if human_mask_full.sum() == 0:
            dist_color_human_full = dist_color_full.copy()
        else:
            hmin, hmax = dist_new_full[human_mask_full].min(), dist_new_full[human_mask_full].max()
            if hmax == hmin:
                # constant map
                tmp = np.full_like(dist_new_full, hmin, dtype=np.float32)
            else:
                tmp = np.clip(dist_new_full, hmin, hmax)
                tmp = ((tmp - hmin) / (hmax - hmin) * 255.0).astype(np.uint8)
            dist_color_human_full = (cmap(tmp.astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Downsample color images for visualization
        dist_color_human_init = cv2.resize(dist_color_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        dist_color_human = cv2.resize(dist_color_human_full, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Downsample tz_map (keeps same behavior as before)
        tz_map_downsampled = cv2.resize(tz_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # If tz_map is 3-channel, ensure shape is restored after resizing
        if tz_map.ndim == 3 and tz_map_downsampled.ndim == 2:
            tz_map_downsampled = tz_map_downsampled[:, :, np.newaxis]

        _img1 = np.concatenate([raw_image_downsampled, pred_rend_array_init, pred_rend_array_bev_init], 1)
        _img2 = np.concatenate([raw_image_downsampled, dist_color_human_init, dist_color_human], 1)
        _img3 = np.concatenate([raw_image_downsampled, pred_rend_array, pred_rend_array_bev], 1)

        _img = np.concatenate([_img1, _img2, _img3], 0)
        # Write mean_height_deviation on the image
        cv2.putText(_img, f'Mean Height Deviation: {mean_height_deviation:.3f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "_".join(img_path.split('/')[-3:])), _img)
    
