import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np
import cv2
import torch
from tqdm import tqdm
from utils.render import render_meshes, render_side_views
from utils.color import demo_color as color
import argparse

class DatasetVisualizer:
    def __init__(self, device='cuda'):
        self.device = device
        self.load_smpl_model()
    
    def load_smpl_model(self):
        from utils.smpl_wrapper import SMPL
        smpl_cfg = {
            "model_path": "data/models/smpl",
            "gender": "neutral",
            "model_type": "smpla",
            "joint_regressor_extra": "data/models/smpl/SMPL_to_J19.pkl"
        }
        self.smpl_model = SMPL(**smpl_cfg).to(self.device)
    
    def visualize_single_image(self, img_path, annots_list, output_path, base_image_dir, resize_size=1024):
        """
        Visualize a single image with its annotations
        
        Args:
            img_path: relative path to image (e.g., 'insta-train/xxx.jpg')
            annots_list: list of person annotations
            output_path: where to save the visualization
            base_image_dir: base directory containing the images
            resize_size: maximum dimension for output image
        """
        # Construct full image path
        full_img_path = os.path.join(base_image_dir, img_path)
        
        if not os.path.exists(full_img_path):
            print(f"Image not found: {full_img_path}")
            return False
        
        # Load image
        raw_image = cv2.imread(full_img_path)
        if raw_image is None:
            print(f"Failed to load image: {full_img_path}")
            return False
        
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        h, w = raw_image.shape[:2]
        
        # Extract data from annotations (support multiple annotation formats)
        verts_list = []
        trans_list = []
        shapes = []
        poses = []
        transl = []

        for person_annot in annots_list:
            smpl_param = person_annot['smpl_param']
            cam_param = person_annot['cam_param']

            shapes.append(smpl_param['shape'])
            poses.append(smpl_param['pose'])
            transl.append(smpl_param['trans'])
        
        if len(shapes) == 0:
            print(f"No person annotations found for {img_path}")
            return False
        
        # Convert to tensors and run SMPL
        shapes_tensor = torch.tensor(np.stack(shapes), dtype=torch.float32).to(self.device)
        poses_tensor = torch.tensor(np.stack(poses), dtype=torch.float32).to(self.device)
        transl_tensor = torch.tensor(np.stack(transl), dtype=torch.float32).to(self.device)
        
        output = self.smpl_model(
            betas=shapes_tensor,
            global_orient=poses_tensor[:, 0:1],
            body_pose=poses_tensor[:, 1:],
            transl=transl_tensor,
        )
        verts_list = output.vertices.detach().cpu().numpy()
        trans_list = transl
        
        # Get camera parameters (same for all people in the image)
        cam_param = annots_list[0]['cam_param']
        focal = cam_param.get('focal')
        princpt = cam_param.get('princpt')
        # if cam_param provided as full K matrix
        if focal is None or princpt is None:
            K = cam_param.get('K') or cam_param.get('cam_int')
            if K is not None:
                K = np.array(K)
                focal = np.array([K[0, 0], K[1, 1]])
                princpt = np.array([K[0, 2], K[1, 2]])
        
        # Construct K matrix
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = focal[0]
        K[1, 1] = focal[1]
        K[0, 2] = princpt[0]
        K[1, 2] = princpt[1]
        
        # Downsample for visualization
        downsample_factor = max(h, w) / resize_size if max(h, w) > resize_size else 1
        new_h, new_w = int(h / downsample_factor), int(w / downsample_factor)
        raw_image_downsampled = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        focal_downsampled = focal / downsample_factor
        princpt_downsampled = princpt / downsample_factor
        K_downsampled = K.copy()
        K_downsampled[:2, :] /= downsample_factor
        
        faces_list = [self.smpl_model.faces for _ in range(len(verts_list))]
        
        pred_rend_array = render_meshes(
            raw_image_downsampled, 
            verts_list, 
            faces_list,
            {'focal': focal_downsampled, 'princpt': princpt_downsampled}, 
            color=color
        )
        
        _, _, pred_rend_array_bev = render_side_views(
            raw_image_downsampled, 
            color, 
            verts_list, 
            faces_list, 
            trans_list, 
            K_downsampled, 
            render_bev=True
        )
        
        # Concatenate images: original, rendered, bird's eye view
        vis_img = np.concatenate([raw_image_downsampled, pred_rend_array, pred_rend_array_bev], axis=1)
        
        # Convert back to BGR for saving
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # Save visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_img)
        
        return

def visualize_dto_dataset(
    npz_path='data/insta/INSTA_CHMR_SMPL_OPT.npz',
    base_image_dir='data/insta/images/insta-train/',
    output_dir='output_vis',
    max_images=10,
    device='cuda'
):
    """
    Visualize the INSTA dataset from the processed npz file
    
    Args:
        npz_path: path to the INSTA_CHMR_SMPL_OPT.npz file
        base_image_dir: base directory containing the insta-train folder
        output_dir: directory to save visualizations
        max_images: maximum number of images to visualize
        device: cuda or cpu
    """
    # Generic loader that supports both {'annots': {...}} and direct per-image keys
    print(f"Loading npz file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # try common layouts
    if 'annots' in data:
        annots_obj = data['annots']
        annots = annots_obj.item() if isinstance(annots_obj, np.ndarray) else annots_obj
    else:
        # build dict from files (each entry may be a dict for an image)
        annots = {}
        for k in data.files:
            try:
                v = data[k]
                annots[k] = v.item() if isinstance(v, np.ndarray) else v
            except Exception:
                annots[k] = data[k]

    print(f"Found {len(annots)} images in the dataset")

    # Initialize visualizer
    visualizer = DatasetVisualizer(device=device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process images
    img_paths = list(annots.keys())
    # uniform sampling: pick max_images indices evenly across the list
    if len(img_paths) > max_images:
        idxs = np.linspace(0, len(img_paths) - 1, num=max_images, dtype=int)
        img_paths = [img_paths[i] for i in idxs]

    for img_path in tqdm(img_paths, desc="Visualizing images"):
        raw_ann = annots[img_path]
        annots_list = raw_ann.item() if isinstance(raw_ann, np.ndarray) else raw_ann

        # Create output path maintaining directory structure
        output_filename = str(img_path).replace('/', '_').replace('\\', '_')
        if not output_filename.endswith('.jpg') and not output_filename.endswith('.png'):
            output_filename += '.jpg'
        output_path = os.path.join(output_dir, output_filename)

        visualizer.visualize_single_image(
            img_path,
            annots_list,
            output_path,
            base_image_dir
        )

def parse_args():
    parser = argparse.ArgumentParser(description='DTO-Humans Visualization')
    parser.add_argument('--dataset', default="coco", help='数据集名称')
    parser.add_argument('--out_dir', default="output_vis", help='输出图像路径')
    parser.add_argument('--vis_num', type=int, default=20, help='可视化数量')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    
    data_path_dict = {
        'insta': ('data/insta/INSTA_CHMR_SMPL_OPT.npz', 'data/insta/images/insta-train/'),
        'coco': ('data/coco2014/COCO_CHMR_SMPL_OPT.npz', 'data/coco2014/images/train2014/'),
        'mpii': ('data/mpii/MPII_CHMR_SMPL_OPT.npz', 'data/mpii/images/'),
        'aic':  ('data/aic/AIC_CHMR_SMPL_OPT.npz', 'data/aic/images/'),
    }

    visualize_dto_dataset(npz_path=data_path_dict[args.dataset][0], base_image_dir=data_path_dict[args.dataset][1], output_dir=args.out_dir, max_images=args.vis_num, device='cuda')
