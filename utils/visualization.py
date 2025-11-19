import os
import io
import torch
import numpy as np
from termcolor import colored
try:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    # os.environ["PYOPENGL_PLATFORM"] = "egl"
    import pyrender
except:
    print(colored('pyrender is not correctly imported.', 'red'))
import matplotlib
from matplotlib import colormaps
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import math
import cv2
import trimesh
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import torchvision
from .transforms import adjust_colors

BASE_COLORS = np.loadtxt(os.path.abspath(os.path.join(__file__, "../colors.txt")), skiprows=0)/255.
BASE_COLORS = adjust_colors(BASE_COLORS,
                            saturation_threshold = 0.3, 
                            brightness_threshold = 0.8)


def get_colors_rgb(size):
    # np.random.seed(131)
    return BASE_COLORS[np.random.choice(BASE_COLORS.shape[0], size=size, replace=False)]

def tensor_to_BGR(img_tensor):
    img = img_tensor.numpy()*255
    img = img.astype(np.uint8).transpose((1,2,0))[:,:,::-1].copy()
    return img

def pad_img(img, pad_size = None, pad_color_offset = 127):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())
    if pad_size is None:
        pad_size = max(img.shape[0],img.shape[1])

    pad = np.zeros((pad_size,pad_size,img.shape[-1]), dtype=img.dtype) + pad_color_offset
    pad[:img.shape[0], :img.shape[1]] = img.copy()
    return pad


def vis_scale_img(img, scale_map, conf_thresh = 0.3, patch_size=14):
    cmap = plt.get_cmap('coolwarm')

    vis_map = np.zeros((scale_map.shape[0]*patch_size, scale_map.shape[1]*patch_size, 3), dtype=np.uint8)
    loc_i, loc_j = torch.where(scale_map[:,:,0] > conf_thresh)
    for (i, j) in zip(loc_i, loc_j):
        scale = round(math.sqrt(scale_map[i,j,1].item()),2)
        vis_map[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] = (np.array(cmap(1-scale)[:3][::-1])*255).astype(np.uint8)
    vis_map = pad_img(vis_map, pad_color_offset=0)
    img = pad_img(img)
    # print(img.shape, vis_map.shape)
    assert img.shape == vis_map.shape

    white_img = 0.6*img + 0.4*np.array((255,255,255))

    valid_mask = (vis_map > 0)
    visible_weight = 0.8
    img = vis_map * valid_mask * visible_weight +\
                img * valid_mask * (1-visible_weight)+\
                white_img * (1-valid_mask)

    # draw patches
    loc_i, loc_j = torch.where(scale_map[:,:,0]+1)
    for (i, j) in zip(loc_i.tolist(), loc_j.tolist()):
        cv2.rectangle(img, (j*patch_size, i*patch_size), ((j+1)*patch_size, (i+1)*patch_size),
                        color=(255,255,255), thickness = 2 )

    return img

def vis_meshes_img(img, verts, smpl_faces, cam_intrinsics, colors = None, padding = True):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())

    if padding:
        pad_size = max(img.shape[0],img.shape[1])
        img = pad_img(img, pad_size)

    if colors is not None:
        assert len(colors) == len(verts)

    if len(cam_intrinsics.flatten()) == 9:
        cam_intrinsics = cam_intrinsics.reshape(3,3)
        rgb, depth = render_mesh(img.shape[0],img.shape[1],verts,smpl_faces,cam_intrinsics,colors)
        valid_mask = (depth > 0)[:,:,None] 
        visible_weight = 1.
        rendered_img = rgb[:,:,::-1] * valid_mask * visible_weight +\
                        img * valid_mask * (1-visible_weight)+\
                        img * (1-valid_mask)
    else:
        rendered_img = img
        for i, cam_int in enumerate(cam_intrinsics):
            rgb, depth = render_mesh(img.shape[0],img.shape[1],[verts[i]],smpl_faces,cam_int,colors)
            valid_mask = (depth > 0)[:,:,None] 
            visible_weight = 0.8
            rendered_img = rgb[:,:,::-1] * valid_mask * visible_weight +\
                            rendered_img * valid_mask * (1-visible_weight)+\
                            rendered_img * (1-valid_mask)
    rendered_img = rendered_img.astype(np.uint8)

    return rendered_img


def vis_joints_img(img, j2ds):
    pass

def vis_sat(img, input_size, patch_size, sat_dict, bid, padding=True):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())

    assert max(img.shape[0], img.shape[1]) == input_size
    if padding:
        img = pad_img(img, input_size)

    # visualize patches
    pos_y, pos_x = sat_dict['pos_y'][bid], sat_dict['pos_x'][bid]
    pos_y = (pos_y * input_size).detach().int().cpu().numpy()
    pos_x = (pos_x * input_size).detach().int().cpu().numpy()

    lvls = sat_dict['lvl']
    if lvls is None:
        lvl = np.zeros(len(pos_x),dtype=int)
    else:
        lvl = lvls[bid].detach().int().cpu().numpy()

    for (cx, cy, l) in zip(pos_x, pos_y, lvl):
        if l == 0:
            half_patch = patch_size//2
            # color = (139, 97, 233)
            color = (173,178,241)
        elif l == 1:
            half_patch = patch_size
            # color = (246, 222, 118)
            color = (239,198,175)
        elif l >= 2:
            half_patch = patch_size*(2**(l-1))
            # color = (0,0,0)
            color = (255, 255, 255)
        else:
            raise NotImplementedError

        x1, x2 = cx - half_patch, cx + half_patch
        y1, y2 = cy - half_patch, cy + half_patch

        if l>0:
            k = 7*l
            img[y1:y2,x1:x2] = 0.5*cv2.blur(img[y1:y2,x1:x2].copy(),(k,k)) + 0.5*np.array(color)
        else:
            # pass
            img[y1:y2,x1:x2] = 0.5*img[y1:y2,x1:x2].copy() + 0.5*np.array(color)


    for (cx, cy, l) in zip(pos_x, pos_y, lvl):
        if l == 0:
            half_patch = patch_size//2
            color = (139, 97, 233)
        elif l == 1:
            half_patch = patch_size
            color = (246, 222, 118)
        elif l >= 2:
            half_patch = patch_size*(2**(l-1))
            color = (255, 255, 255)
        else:
            raise NotImplementedError
        
        x1, x2 = cx - half_patch, cx + half_patch
        y1, y2 = cy - half_patch, cy + half_patch

        cv2.rectangle(img, (x1, y1), (x2, y2),
            color=(255,255,255), thickness = 2 )
        

    return img

def vis_boxes(img, boxes, padding=True, color = (0,0,255)):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())
    if padding:
        pad_size = max(img.shape[0],img.shape[1])
        img = pad_img(img, pad_size)
    
    for bbox in boxes:
        bbox = bbox.int().tolist()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
            color=color, thickness = 2 )
    
    return img


def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=False, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def render_mesh(height, width, meshes, face, cam_intrinsics, colors = None):
    
    # renderer
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # mesh
    if colors is None:
        colors = get_colors_rgb(len(meshes))

    for i, mesh in enumerate(meshes):
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(*colors[i], 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

        scene.add(mesh, f'mesh_{i}')


    # camera
    f=np.array([cam_intrinsics[0,0],cam_intrinsics[1,1]])
    c=cam_intrinsics[0:2,2]
    camera = pyrender.camera.IntrinsicsCamera(fx=f[0], fy=f[1], cx=c[0], cy=c[1])
    scene.add(camera)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    renderer.delete()
    return rgb, depth
