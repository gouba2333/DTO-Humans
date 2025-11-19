import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch

from core.utils import recursive_to
from core.datasets.dataset import Dataset
from core.utils.geometry import batch_rot2aa
from mesh_estimator import HumanMeshEstimator
from smplx.lbs import vertices2joints
from utils.visualization import vis_meshes_img

from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy
from core.constants import DETECTRON_CFG, DETECTRON_CKPT

def get_args():
    parser = argparse.ArgumentParser(description="Generate pseudo ground truth")
    parser.add_argument("--image_folder", type=str, default="demo")
    parser.add_argument("--annots_path", type=str, default=None)
    parser.add_argument("--chmr_file_path", type=str, default="data/demo.npz")
    parser.add_argument("--det_threshold", type=float, default=0.5, help="Detection threshold for bbox filtering")
    parser.add_argument("--det_nms_threshold", type=float, default=0.7, help="Threshold for NMS")
    parser.add_argument("--IoU_threshold", type=float, default=0.5, help="IoU threshold for bbox matching")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device to use")
    args = parser.parse_args()
    return args

def nms_kp2d(kp2ds, scores, boxes, dist_ratio=1/8, vis_thresh: float = 0.5) -> np.ndarray:
    if len(kp2ds) == 0:
        return np.array([], dtype=np.int64)

    has_vis = (kp2ds.ndim == 3 and kp2ds.shape[2] == 3)

    order = np.argsort(-scores)  # desc by score
    keep = []
    suppressed = np.zeros(len(kp2ds), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        base_box = boxes[idx]
        base_side = max(base_box[2] - base_box[0], base_box[3] - base_box[1])
        dist_thresh = dist_ratio * base_side * 0.6

        # base keypoints (x,y) and visibility
        if has_vis:
            base_kpts_xy = kp2ds[idx][:, :2]
            base_vis = kp2ds[idx][:, 2] >= vis_thresh
        else:
            base_kpts_xy = kp2ds[idx]
            base_vis = np.ones(base_kpts_xy.shape[0], dtype=bool)

        for j in order:
            if j == idx or suppressed[j]:
                continue

            if has_vis:
                comp_kpts_xy = kp2ds[j][:, :2]
                comp_vis = kp2ds[j][:, 2] >= vis_thresh
                common = base_vis & comp_vis
                if common.sum() > 0:
                    dist = np.linalg.norm(base_kpts_xy[common] - comp_kpts_xy[common], axis=1).mean()
                else:
                    # fallback: use mean distance over all keypoints
                    dist = np.linalg.norm(base_kpts_xy - comp_kpts_xy, axis=1).mean()
            else:
                comp_kpts_xy = kp2ds[j]
                dist = np.linalg.norm(base_kpts_xy - comp_kpts_xy, axis=1).mean()

            if dist < dist_thresh:
                suppressed[j] = True
                continue

    return np.array(keep, dtype=np.int64)

class Pipeline:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.estimator = HumanMeshEstimator(device=device)

        print("Loading Detectron2...")
        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = args.det_threshold
            detectron2_cfg.model.roi_heads.box_predictors[i].test_nms_thresh = args.det_nms_threshold
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)

    def inference(self, image_folder, annots_path=None):
        results = {}
        
        if annots_path is not None:
            annots = np.load(annots_path, allow_pickle=True)["annots"].item()
            img_names = list(annots.keys())
        else:
            img_names = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        for img_name in tqdm(img_names, ncols=100):
            img_path = os.path.join(image_folder, img_name)
            img_cv2 = cv2.imread(img_path)
            if img_cv2 is None:
                print(f"[WARN] Load image failed: {img_path}")
                continue
            h, w, _ = img_cv2.shape

            # ====== 1. Human detection ======
            det_out = self.detector(img_cv2)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > self.args.det_threshold)
            confs = det_instances.scores[valid_idx].cpu().numpy()
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            if len(boxes) == 0:
                print("Skipping image with no person:", img_name)
                continue

            bbox_centers, bbox_scales = [], []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
                size = np.max([x2 - x1, y2 - y1])
                bbox_centers.append(center)
                bbox_scales.append(size / 200.0)
            bbox_centers = np.array(bbox_centers)
            bbox_scales = np.array(bbox_scales)

            # ====== 2. Camera intrinsics ======
            cam_int = self.estimator.get_cam_intrinsics(img_cv2)   # torch.Tensor shape (3, 3)
            if not isinstance(cam_int, torch.Tensor):
                cam_int = torch.tensor(cam_int, dtype=torch.float32, device=self.device)

            # ====== 3. Dataset organization ======
            dataset = Dataset(img_cv2, bbox_centers, bbox_scales, cam_int, False, img_path)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

            all_pose = []
            all_shape = []
            all_trans = []
            all_kp2ds = []

            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out_smpl_params, out_cam, focal_length_ = self.estimator.model(batch)
                output_vertices, output_joints, output_cam_trans = self.estimator.get_output_mesh_with_transl(
                    out_smpl_params, out_cam, batch
                )

                # ====== 4. joints & kp2ds projection ======
                J_regressor_extra = np.load('data/models/smpl/J_regressor_extra.npy')
                J_regressor_extra = torch.tensor(J_regressor_extra, dtype=torch.float32).to(self.device)
                model_joints_extra = vertices2joints(J_regressor_extra, output_vertices)

                model_joints = torch.concat([output_joints[:, :24], model_joints_extra], dim=1) # 24+9
                model_joints = model_joints[:, [16,17,18,19,20,21,1,2,4,5,7,8,27,26], :]  # 14 keypoints for CrowdPose
                # Project to 2D
                pred_j2ds_homo = torch.einsum('bjc,cd->bjd', model_joints, cam_int.transpose(0, 1))
                pred_j2ds = pred_j2ds_homo[..., :] / (pred_j2ds_homo[..., 2:] + 1e-6)

                # ====== 5. SMPL parameter conversion ======
                pose = torch.cat([out_smpl_params["global_orient"], out_smpl_params["body_pose"]], dim=1)
                b, j, c, d = pose.shape
                pose = batch_rot2aa(pose.view(b * j, c, d)).view(b, -1).cpu().numpy()
                shape = out_smpl_params["betas"].cpu().numpy()
                trans = output_cam_trans.cpu().numpy()
                kp2ds = pred_j2ds.cpu().numpy()

                all_pose.append(pose)
                all_shape.append(shape)
                all_trans.append(trans)
                all_kp2ds.append(kp2ds)

            # Concatenate into (N_person, xx)
            all_pose = np.concatenate(all_pose, axis=0)
            all_shape = np.concatenate(all_shape, axis=0)
            all_trans = np.concatenate(all_trans, axis=0)
            all_kp2ds = np.concatenate(all_kp2ds, axis=0)

            # nms by kp2ds
            keep_inds = nms_kp2d(all_kp2ds, scores=confs, boxes=boxes, dist_ratio=1.0/8.0)
            if len(keep_inds) < len(all_kp2ds):
                all_pose = all_pose[keep_inds]
                all_shape = all_shape[keep_inds]
                all_trans = all_trans[keep_inds]
                all_kp2ds = all_kp2ds[keep_inds]
            cam_int = cam_int.unsqueeze(0).repeat(len(all_pose), 1, 1).cpu().numpy()
            
            results[img_name] = {
                "trans_cam": all_trans,         # (N, 3)
                "pose_cam": all_pose,           # (N, 72)
                "shape": all_shape,             # (N, 10)
                "cam_int": cam_int,             # (N, 3, 3)
            }
        
        return results
    

if __name__ == "__main__":
    args = get_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    pipeline = Pipeline(device=device, args=args)

    results = pipeline.inference(args.image_folder, args.annots_path)
    np.savez(args.chmr_file_path, annots=results)
    print(f"Saved results to {args.chmr_file_path}")
