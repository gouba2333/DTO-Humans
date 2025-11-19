import os
import argparse
import numpy as np
import time
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
from torchvision import transforms

from inference_init import nms_kp2d
from mivolo.model.yolo_detector import Detector
from mivolo.data.misc import assign_faces

from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor
from transformers import AutoProcessor, VitPoseForPoseEstimation
import cv2
import pickle
from dto import Optimizer
from tqdm import tqdm

from core.utils import recursive_to
from core.utils.geometry import batch_rot2aa
from core.datasets.dataset import Dataset
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy

from core.camerahmr_model import CameraHMR
from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, DETECTRON_CKPT, DETECTRON_CFG
from utils.evaluation import calculate_iou
import ast

from utils.smpl_wrapper import SMPL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_annot(annotation, fields, dir_depth):
    imgnames = annotation["imgname"]
    unique_imgnames, inverse_indices = np.unique(imgnames, return_inverse=True)

    group_indices = defaultdict(list)
    for i, img_idx in enumerate(inverse_indices):
        group_indices[img_idx].append(i)

    field_data = {f: np.array(annotation[f]) for f in fields if len(annotation[f]) > 0} 

    annot_by_img = {}
    for img_idx, indices in group_indices.items():
        img_name = '/'.join(unique_imgnames[img_idx].split('/')[-dir_depth:])
        img_data = {}
        indices_arr = np.array(indices) 

        for field in field_data.keys():
            img_data[field] = field_data[field][indices_arr]
        annot_by_img[str(img_name)] = img_data
    return annot_by_img

def convert_to_full_img_cam(pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    tz = 2. * focal_length / (bbox_height * s)
    cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t

def get_output_mesh_with_transl(smpl_model, params, pred_cam, batch):
    img_h, img_w = batch['img_size'][0]
    cam_trans = convert_to_full_img_cam(
        pare_cam=pred_cam,
        bbox_height=batch['box_size'],
        bbox_center=batch['box_center'],
        img_w=img_w,
        img_h=img_h,
        focal_length=batch['cam_int'][:, 0, 0]
    )
    params['transl'] = cam_trans
    params['pose2rot'] = False
    smpl_output = smpl_model(**{k: v for k, v in params.items()})
    pred_keypoints_3d = smpl_output.joints
    pred_vertices = smpl_output.vertices
    
    return pred_vertices, pred_keypoints_3d, cam_trans

def parse_args():
    parser = argparse.ArgumentParser(description='Single-image inference script')
    parser.add_argument('--det_threshold', type=float, default=0.5, help='Human detection confidence threshold')
    parser.add_argument('--det_nms_threshold', type=float, default=0.7, help='NMS IoU threshold for detection')
    parser.add_argument('--human_face_detector_weights', default="ckpt/yolov8x_person_face.pt", help='Path to YOLO face detector weights')
    parser.add_argument('--dataset', default="demo", help='Dataset name')
    parser.add_argument('--out_dir', default="output", help='Directory for output images')
    parser.add_argument('--vis_step', type=int, default=-1, help='Visualization interval (frames)')
    parser.add_argument('--use_yolo', type=ast.literal_eval, default=True, help='Use YOLO for more accurate face boxes')
    parser.add_argument('--use_smpla', type=ast.literal_eval, default=True, help='Use SMPL+A model for pose estimation')
    parser.add_argument('--save_all_annots', type=ast.literal_eval, default=True, help='Save all annotations')
    parser.add_argument('--use_age_prior', type=ast.literal_eval, default=True, help='Use age prior')
    parser.add_argument('--use_gender_prior', type=ast.literal_eval, default=True, help='Use gender prior')
    parser.add_argument('--use_X_cond', type=ast.literal_eval, default=True, help='Use internal body depth constraints')
    parser.add_argument('--skip_optimize', type=ast.literal_eval, default=False, help='Skip optimization; only fill seg-based CHMR annotations')
    parser.add_argument('--start_sample', type=int, default=-1, help='Start sample index for optimization')
    parser.add_argument('--end_sample', type=int, default=-1, help='End sample index for optimization')
    parser.add_argument('--out_npz_postname', '-n', default="", help='Suffix for output NPZ filename')
    return parser.parse_args()

def main():
    args = parse_args()
    # annotation path, image root path, dir_depth
    data_path_dict = {
        "insta1": ("data/insta/insta1-release.npz", "data/insta/images/insta-train/", 3),
        "insta2": ("data/insta/insta2-release.npz", "data/insta/images/insta-train/", 3),
        "coco": ("data/coco2014/coco-release.npz", "data/coco2014/images/train2014", 1),
        "aic": ("data/aic/aic-release.npz", "data/aic/images/", 1),
        "mpii": ("data/mpii/mpii-release.npz", "data/mpii/images/", 1),
        "rh": ("data/relativehuman_test.npz", "data/RelativeHuman/images/", 1),
        "demo": ("data/demo.npz", "demo/", 1),
    }

    annot_filename = data_path_dict[args.dataset][0]
    print(f"Loading annotations from {annot_filename}")

    optimize_num = 0
    try:
        annots = np.load(annot_filename, allow_pickle=True)["annots"][()]
        annots = {key: annots[key] for key in annots}
    except Exception as e:
        annots = np.load(annot_filename, allow_pickle=True)
        annots = {key: annots[key][()] for key in annots.files}

    imgnames = annots["imgname"] if "imgname" in annots.keys() else list(annots.keys())
    if len(imgnames[0].split('/')) != data_path_dict[args.dataset][2]:
        imgnames = ["/".join(imgname.split('/')[-data_path_dict[args.dataset][2]:]) for imgname in imgnames]
    unique_imgnames, inverse_indices = np.unique(imgnames, return_inverse=True)

    if args.start_sample >= 0:
        unique_imgnames = unique_imgnames[args.start_sample:args.end_sample]
    print(len(unique_imgnames))

    if 'shape' in annots:  # annotations are instance-level
        if 'pose_cam' not in annots:
            annots['pose_cam'] = annots['pose']
            del annots['pose']
        if 'trans_cam' not in annots:
            annots['trans_cam'] = annots['cam_t']
            del annots['cam_t']

        fields = list(annots.keys())
        annots = convert_annot(annots, fields, dir_depth=data_path_dict[args.dataset][2])

    if args.use_yolo:
        print("Loading YOLO for face detection...")
        human_face_detector = Detector(args.human_face_detector_weights, device)

    # Setup MiVOLO v2 (Hugging Face)
    print("Loading MiVOLO v2...")
    mivolo_repo = "iitolstykh/mivolo_v2"
    torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model_mivolo = AutoModelForImageClassification.from_pretrained(
        mivolo_repo, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device)
    image_processor = AutoImageProcessor.from_pretrained(mivolo_repo, trust_remote_code=True)
    
    print("Loading Detectron2...")
    detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
    detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = args.det_threshold
        detectron2_cfg.model.roi_heads.box_predictors[i].test_nms_thresh = args.det_nms_threshold
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    print("Loading CameraHMR...")
    hmr_model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
    hmr_model = hmr_model.to(device)
    hmr_model.eval()

    smpl_cfg = {
        "model_path": "data/models/smpl",
        "gender": "neutral",
        "model_type": "smpl",
        "joint_regressor_extra": "data/models/smpl/SMPL_to_J19.pkl"
    }
    smpl_model = SMPL(**smpl_cfg).to(device)
    args.use_age_prior = False if args.dataset in ['mupots'] else args.use_age_prior
    optimizer = Optimizer(device=device, args=args)

    # Setup ViTPose (Hugging Face)
    print("Loading ViTPose...")
    vitpose_image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    vitpose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device)
    
    annotations = {}
    time_stats = {
        '1_det': [],
        '2_hmr': [],
        '3_mivolo_yolo': [],
        '4_mivolo_infer': [],
        '5_opt': [],
    }

    for img_name in tqdm(unique_imgnames, ncols=100):
        if str(img_name) in annots:
            exist_annot = annots[str(img_name)]
        else:
            print(img_name, 'not in annots')
            continue
        if 'pose_cam' not in exist_annot:
            exist_annot['pose_cam'] = exist_annot['pose']
            del exist_annot['pose']
        if 'trans_cam' not in exist_annot:
            exist_annot['trans_cam'] = exist_annot['trans']
            del exist_annot['trans']
        annot_sup = {}
        img_path = os.path.join(data_path_dict[args.dataset][1], img_name)
        img_cv2 = cv2.imread(str(img_path))
        H, W, _ = img_cv2.shape

        # ====== 1. Human detection ======
        t1 = time.time()
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > args.det_threshold)
        confs = det_instances.scores[valid_idx].cpu().numpy()
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        masks = det_instances.pred_masks[valid_idx].cpu().numpy()
        cam_int = exist_annot['cam_int'][0]
        if not isinstance(cam_int, torch.Tensor):
            cam_int = torch.tensor(cam_int, dtype=torch.float32, device=device)

        # ====== 1.5 ViTPose keypoints — filter detections with few visible keypoints ======
        if len(boxes) > 0:
            pil_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            boxes_xywh = boxes.copy()
            boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
            boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]
            # vitpose_inputs = vitpose_image_processor([pil_image], boxes=[boxes_xywh], return_tensors="pt").to(device)
            # use_mask
            vitpose_inputs = vitpose_image_processor([pil_image*masks[i][:, :, None] for i in range(len(masks))], boxes=[box[np.newaxis, ...] for box in boxes_xywh], return_tensors="pt").to(device)

            with torch.no_grad():
                vitpose_outputs = vitpose_model(**vitpose_inputs)
            vitpose_results = vitpose_image_processor.post_process_pose_estimation(vitpose_outputs, boxes=[boxes_xywh])
            # Extract keypoints and scores for all detections from ViTPose
            vitpose_kpts_all = []
            vitpose_scores_all = []
            for person in vitpose_results[0]:
                # person['keypoints']: tensor (K,3) ; person['scores']: tensor (K,)
                kpts = person['keypoints'].cpu().numpy()
                scores = person['scores'].cpu().numpy()
                vitpose_kpts_all.append(kpts)
                vitpose_scores_all.append(scores)
            # Filter out boxes with fewer than 5 visible keypoints (score >= 0.5)
            keep_inds = []
            for i, scores in enumerate(vitpose_scores_all):
                num_visible = (scores >= 0.5).sum()
                if num_visible >= 5:
                    keep_inds.append(i)
            if len(keep_inds) == 0:
                print(f"[WARN] No valid human after ViTPose filter: {img_path}")
                continue
            boxes = boxes[keep_inds]
            confs = confs[keep_inds]
            masks = masks[keep_inds]
            vitpose_kpts_all = np.array([vitpose_kpts_all[i] for i in keep_inds])
            vitpose_scores_all = np.array([vitpose_scores_all[i] for i in keep_inds])
        else:
            print(f"[WARN] No human detected: {img_path}")
            continue

        # ====== NMS to remove duplicate human predictions ======
        keep_inds = nms_kp2d(np.concatenate((vitpose_kpts_all, vitpose_scores_all[:, :, np.newaxis]), axis=2), scores=confs, boxes=boxes, dist_ratio=1.0/8.0)
        if len(keep_inds) < len(vitpose_kpts_all):
            boxes = boxes[keep_inds]
            masks = masks[keep_inds]
            confs = confs[keep_inds]
            vitpose_kpts_all = [vitpose_kpts_all[i] for i in keep_inds]
            vitpose_scores_all = [vitpose_scores_all[i] for i in keep_inds]

        # ====== 2. Human mesh estimation ======
        t2 = time.time()
        bbox_centers, bbox_scales = [], []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
            size = np.max([x2 - x1, y2 - y1])
            bbox_centers.append(center)
            bbox_scales.append(size / 200.0)
        bbox_centers = np.array(bbox_centers)
        bbox_scales = np.array(bbox_scales)

        dataset = Dataset(img_cv2, bbox_centers, bbox_scales, cam_int, False, img_path, masks, use_mask=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = hmr_model(batch)
            output_vertices, output_joints, output_cam_trans = get_output_mesh_with_transl(
                smpl_model, out_smpl_params, out_cam, batch
            )

            pred_j2ds_homo = torch.einsum('bjc,cd->bjd', output_joints, cam_int.transpose(0, 1))
            pred_j2ds = pred_j2ds_homo[..., :2] / (pred_j2ds_homo[..., 2:] + 1e-6)

            pose = torch.cat([out_smpl_params["global_orient"], out_smpl_params["body_pose"]], dim=1)
            b, j, c, d = pose.shape
            pose = batch_rot2aa(pose.view(b * j, c, d)).view(b, -1).cpu().numpy()
            shape = out_smpl_params["betas"].cpu().numpy()
            trans = output_cam_trans.cpu().numpy()
            kp2ds = pred_j2ds.cpu().numpy()
        
        if "gtkps" in exist_annot:
            kp2ds_chmr = exist_annot['gtkps'][:, :44, :2] # K, 44, 2
        else:
            cam_int = exist_annot['cam_int'][0]
            kpts_3d = smpl_model(
                betas = torch.tensor(exist_annot['shape']).to(device),
                global_orient = torch.tensor(exist_annot['pose_cam'][:, :3]).to(device),
                body_pose = torch.tensor(exist_annot['pose_cam'][:, 3:]).to(device),
                transl = torch.tensor(exist_annot['trans_cam']).to(device),
            ).joints
            kp2ds_chmr = torch.einsum('bjc,cd->bjd', kpts_3d, torch.tensor(cam_int).transpose(0, 1).to(device)).cpu().numpy() # K, 44, 2
            kp2ds_chmr = kp2ds_chmr[..., :2] / (kp2ds_chmr[..., 2:] + 1e-6)

        # ====== 3. Matching ====== 
        # threshold from keypoints
        def head_size_from_kpts(kpts):
            return np.linalg.norm(kpts[38] - kpts[37])

        def dist_thresh_from_kpts(kpts):
            return 0.6 * head_size_from_kpts(kpts)

        N_v = len(vitpose_kpts_all)
        N_p = len(kp2ds)
        vit_used = []
        gt_used = []
        pred_used = []

        # A: vitpose -> gt
        for v_idx, (v_kpts, v_scores) in enumerate(zip(vitpose_kpts_all, vitpose_scores_all)):
            vis_idx = np.where(v_scores >= 0.5)[0]
            if len(vis_idx) == 0:
                continue
            best_gt = -1
            best_score = -1
            for g_idx, gt_kpts in enumerate(kp2ds_chmr):
                if g_idx in gt_used:
                    continue
                gt_kps_cal = gt_kpts[[0, 16, 15, 18, 17, 34, 33, 35, 32, 36, 31, 28, 27, 29, 26, 30, 25]]
                dists = np.linalg.norm(v_kpts[:, :2] - gt_kps_cal[:, :2], axis=1)[vis_idx]
                match_cnt = (dists <= dist_thresh_from_kpts(gt_kpts)).sum()
                match_ratio = match_cnt / len(vis_idx)
                if match_ratio > 0.5: # More than half of the points match
                    if match_cnt > best_score:
                        best_score = match_cnt
                        best_gt = g_idx
            if best_gt >= 0:
                gt_used.append(best_gt)
                vit_used.append(v_idx)

        # B: unmatched vitpose -> match to predicted kp2ds (vit as reference)
        vit_unmatched = [i for i in range(N_v) if i not in vit_used]
        for v_idx in vit_unmatched:
            v_kpts = vitpose_kpts_all[v_idx]
            v_scores = vitpose_scores_all[v_idx]
            vis_idx = np.where(v_scores >= 0.5)[0]
            if len(vis_idx) == 0:
                continue
            best_p = -1
            best_score = -1
            for p_idx in range(N_p):
                if p_idx in pred_used:
                    continue
                p_kpts = kp2ds[p_idx]
                p_kps_cal = p_kpts[[0, 16, 15, 18, 17, 34, 33, 35, 32, 36, 31, 28, 27, 29, 26, 30, 25]]
                dists = np.linalg.norm(v_kpts[:, :2] - p_kps_cal[:, :2], axis=1)[vis_idx]
                match_cnt = (dists <= dist_thresh_from_kpts(p_kpts)).sum()
                match_ratio = match_cnt / len(vis_idx)
                if match_ratio > 0.5: # More than half of the points match
                    if match_cnt > best_score:
                        best_score = match_cnt
                        best_p = p_idx
            if best_p >= 0:
                pred_used.append(best_p)
        
        if len(gt_used) + len(pred_used) == 0:
            print(f"[WARN] No valid human after matching: {img_path}")
            continue

        # ====== 4. Assemble annot_sup ======
        annot_sup = {'conf': [], 'bbox': [], 'pose_cam': [], 'shape': [], 'trans_cam': [],
                     'seg': [], 'kpts_2d': [], 'is_released': [], 'cam_int': []}

        # add matched GTs (use exist_annot pose/shape/trans), prefer vit matches
        for vit_id, gt_id in zip(vit_used, gt_used):
            annot_sup['conf'].append(confs[vit_id])
            annot_sup['bbox'].append(boxes[vit_id])
            annot_sup['pose_cam'].append(exist_annot['pose_cam'][gt_id].flatten())
            annot_sup['shape'].append(exist_annot['shape'][gt_id])
            annot_sup['trans_cam'].append(exist_annot['trans_cam'][gt_id])
            annot_sup['seg'].append(masks[vit_id])
            annot_sup['kpts_2d'].append(kp2ds_chmr[gt_id])
            annot_sup['cam_int'].append(exist_annot['cam_int'][0])
            annot_sup['is_released'].append(True)
        for p_idx in pred_used:
            annot_sup['conf'].append(confs[p_idx])
            annot_sup['bbox'].append(boxes[p_idx])
            annot_sup['pose_cam'].append(pose[p_idx].flatten())
            annot_sup['shape'].append(shape[p_idx])
            annot_sup['trans_cam'].append(trans[p_idx])
            annot_sup['seg'].append(masks[p_idx])
            annot_sup['kpts_2d'].append(kp2ds[p_idx])
            annot_sup['cam_int'].append(exist_annot['cam_int'][0])
            annot_sup['is_released'].append(False)

        # ----- 5. Use MiVOLO v2 to predict gender and age -----
        t3 = time.time()
        all_body_crops = [] # Store raw np.array body crops
        all_face_crops = [] # Store raw np.array face crops
        # Pre-calculate face regions for all people (estimated using keypoints)
        est_face_bboxes = []
        for i in range(len(annot_sup['bbox'])):
            kpts = annot_sup['kpts_2d'][i]
            face_kpts = kpts[[0, 15, 16, 17, 18], :]
            if face_kpts[1, 0] > face_kpts[2, 0]: # Face is backward
                est_face_bboxes.append(torch.tensor([0, 0, 0, 0]))
                continue
            min_x, max_x = face_kpts[:, 0].min(), face_kpts[:, 0].max()
            min_y, max_y = face_kpts[:, 1].min(), face_kpts[:, 1].max()
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            face_width = max_x - min_x
            box_size = face_width * 1.2
            fx1 = int(center_x - box_size / 2)
            fy1 = int(center_y - box_size / 2)
            fx2 = int(center_x + box_size / 2)
            fy2 = int(center_y + box_size / 2)
            est_face_bboxes.append(torch.tensor([fx1, fy1, fx2, fy2]))
        if args.use_yolo:
            # ----- 5.5 Use YOLO for face detection -----
            with torch.no_grad():
                human_face_detector_result = human_face_detector.predict_batch(img_cv2)
            face_bboxes = [human_face_detector_result[0].get_bbox_by_ind(ind).cpu() for ind in human_face_detector_result[0].get_bboxes_inds("face")]
            # Match estimated face regions with YOLO face bboxes
            face_idx, _ = assign_faces(face_bboxes, est_face_bboxes, iou_thresh=0.3)
        
        t4 = time.time()
        for i in range(len(annot_sup['conf'])):
            # --- 1. Get Body Crop (as np.ndarray) ---
            x1, y1, x2, y2 = annot_sup['bbox'][i].astype(int)
            body_x1, body_y1 = max(0, x1), max(0, y1)
            body_x2, body_y2 = max(0, min(W, x2)), max(0, min(H, y2))
            body_crop = img_cv2[body_y1:body_y2, body_x1:body_x2]
            # body_crop = (img_cv2*annot_sup['seg'][i][:, :,None])[body_y1:body_y2, body_x1:body_x2]
            all_body_crops.append(body_crop)

            # --- 2. Get Face Crop (as np.ndarray) ---
            if args.use_yolo and face_idx[i] is not None:
                fx1, fy1, fx2, fy2 = face_bboxes[face_idx[i]]
                face_x1, face_y1 = max(0, fx1), max(0, fy1)
                face_x2, face_y2 = max(0, min(W, fx2)), max(0, min(H, fy2))
                all_face_crops.append(img_cv2[face_y1:face_y2, face_x1:face_x2])
            else:
                # Directly use pre-calculated est_face_bboxes
                fx1, fy1, fx2, fy2 = est_face_bboxes[i].tolist()
                face_x1, face_y1 = max(0, fx1), max(0, fy1)
                face_x2, face_y2 = max(0, min(W, fx2)), max(0, min(H, fy2))
                face_crop = img_cv2[face_y1:face_y2, face_x1:face_x2]
                all_face_crops.append(face_crop if face_crop.size > 0 else None)
                # all_face_crops.append(None)

        def flip_imgs(imgs):
            return [cv2.flip(img, 1) if img is not None and img.size > 0 else img for img in imgs]

        # Concatenate normal and flipped images
        faces_all = all_face_crops + flip_imgs(all_face_crops)
        bodies_all = all_body_crops + flip_imgs(all_body_crops)

        faces_input = image_processor(images=faces_all)["pixel_values"]
        body_input = image_processor(images=bodies_all)["pixel_values"]
        faces_input = faces_input.to(dtype=model_mivolo.dtype, device=device)
        body_input = body_input.to(dtype=model_mivolo.dtype, device=device)

        with torch.no_grad():
            output = model_mivolo(faces_input=faces_input, body_input=body_input)

        n = len(all_body_crops)
        ages = output.age_output.detach().float().cpu().numpy()
        ages = (ages[:n] + ages[n:]) / 2.0
        gender_probs = output.gender_probs.detach().float().cpu().numpy()
        gender = output.gender_class_idx.detach().float().cpu().numpy()
        gender = np.where(gender < 0.5, gender_probs, 1 - gender_probs)
        gender = (gender[:n] + gender[n:]) / 2.0

        annot_sup['age'] = ages
        annot_sup['gender'] = gender

        # Convert to array
        for key in annot_sup.keys():
            annot_sup[key] = np.array(annot_sup[key])

        # ----- 6. Optimize images -----           
        t5 = time.time() 
        imgname = "/".join(img_path.split('/')[-data_path_dict[args.dataset][2]:])

        
        new_annot, is_optimize, mean_height_deviation = optimizer.optimize(img_cv2, img_path, annot_sup, output_dir='output', vis=args.vis_step>0, skip_optimize=args.skip_optimize)
        is_optimize = is_optimize and mean_height_deviation < 1.5
        if is_optimize:
            optimize_num += 1
        if args.save_all_annots or is_optimize:
            annotations[imgname] = new_annot
            
        t6 = time.time()
        time_stats['1_det'].append(t2 - t1)
        time_stats['2_hmr'].append(t3 - t2)
        time_stats['3_mivolo_yolo'].append(t4 - t3)
        time_stats['4_mivolo_infer'].append(t5 - t4)
        time_stats['5_opt'].append(t6 - t5)

    print(f"Total images: {len(unique_imgnames)}, Optimized images: {optimize_num}")
    for key in time_stats.keys():
        print(f"{key}: {np.mean(time_stats[key]):.4f}s")

    save_path = annot_filename.replace('.pkl', '.npz').replace('.npz', '_opt.npz')
    
    if args.out_npz_postname != "":
        save_path = save_path.replace('.npz', f'_{args.out_npz_postname}.npz')

    print(f"Saving results to {save_path} ...")
    np.savez(save_path, **annotations)
    if args.dataset == 'rh':
        from eval_rh import evaluate_rh
        evaluate_rh(results_path=save_path)

if __name__ == '__main__':
    main()
