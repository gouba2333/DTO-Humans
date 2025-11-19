"""
Modified from https://github.com/Arthur151/ROMP/tree/master/trace/lib/evaluation/mupots_util
"""
import numpy as np


def procrustes(predicted, target):
    predicted = predicted.T 
    target = target.T
    predicted = predicted[None, ...]
    target = target[None, ...]

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    return predicted_aligned[0].T


def match(pose1, pose2, threshold=250):
    # Find the matchest p2 in pose2 with each p1 in pose1
    matches = []
    p2 = np.float32(pose2)
    p2 = p2 - p2[:, :, :1]
    for i in range(len(pose1)):
        p1 = np.float32(pose1[i])
        p1 = p1 - p1[:, :1]
        diffs = []
        for j in range(len(p2)):
            p = p2[j]
            p = procrustes(p, p1)
            diff = np.sqrt(np.power(p - p1, 2).sum(axis=0)).mean()
            diffs.append(diff)
        diffs = np.float32(diffs)
        idx = np.argmin(diffs).item()
        if diffs.min() > threshold:
            matches.append(-1)
        else:
            matches.append(idx)
    return matches

# Parents of joints in MuPoTS joint set
_JOINT_PARENTS = np.array([2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2]) - 1
# The order in which joints are scaled, from the hip to outer limbs
_TRAVERSAL_ORDER = np.array([15, 16, 2, 1, 17, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) - 1

def _match_poses(gt_pose_2d, gt_visibility, pred_pose_2d, pred_visibility, threshold, verbose=False):
    """
    Implements ``mpii_multiperson_get_identity_matching.m``.

    Parameters:
        gt_pose_2d: (nGtPoses,nJoints,2), ground truth 2D poses on the image.
        gt_visibility: (nGtPoses,nJoints), True if the given joint is visible in ground-truth.
        gt_pose_2d: (nPredPoses,nJoints,2) predicted 2D poses on the image.
        pred_visibility: (nPredPoses,nJoints), True if the given joint is visible in predictions.

    Returns:
        ndarray(nGtPoses), the indices of the matched predicted pose for all ground truth poses. If no
        matches were found, the value is -1.
    """
    pair_ind = -np.ones(len(gt_pose_2d), dtype='int64')  # -1 means no pair, otherwise the pair id
    has_gt_pair = np.zeros(len(pred_pose_2d), dtype='bool')  # True means the predicted pose is already matched up

    if verbose:
        print(gt_visibility)
        print(pred_visibility)

    for i in range(len(gt_pose_2d)):
        diff = np.abs(gt_pose_2d[[i]] - pred_pose_2d)  # (nPredPose, nJoints, 2)
        matches = np.all(diff < threshold, axis=2)  # (nPredPose, nJoints)
        match_scores = np.sum(matches * (gt_visibility[[i]] & pred_visibility), axis=1)
        match_scores[has_gt_pair] = 0  # zero out scores for already matched up pred_poses

        if verbose:
            print(match_scores)

        best_match_ind = np.argmax(match_scores)
        if match_scores[best_match_ind] > 0:
            pair_ind[i] = best_match_ind
            has_gt_pair[best_match_ind] = True

    return pair_ind

def _scale_to_gt(pred_poses, gt_poses):
    """ Scales bone lengths in pred_poses to match gt_poses. Corresponds to ``mpii_map_to_gt_bone_lengths.m``."""
    rescaled_pred_poses = pred_poses.copy()

    for ind in _TRAVERSAL_ORDER:
        parent = _JOINT_PARENTS[ind]
        gt_bone_length = np.linalg.norm(gt_poses[ind] - gt_poses[parent])
        pred_bone = pred_poses[ind] - pred_poses[parent]
        pred_bone = pred_bone * gt_bone_length / \
                    (np.linalg.norm(pred_bone) + 1e-8)
        rescaled_pred_poses[ind] = rescaled_pred_poses[parent] + pred_bone

    return rescaled_pred_poses

def mean(l):
    return sum(l) / len(l)


def h36m_joint_groups():
    joint_groups = [
        ['Head', [10]],
        ['Neck', [8]],
        ['Shou', [11, 14]],
        ['Elbow', [12, 15]],
        ['Wrist', [13, 16]],
        ['Hip', [1, 4]],
        ['Knee', [2, 5]],
        ['Ankle', [3, 6]],
    ]
    all_joints = []
    for i in joint_groups:
        all_joints += i[1]
    return joint_groups, all_joints

def mpii_joint_groups():
    joint_groups = [
        ['Head', [0]],
        ['Neck', [1]],
        ['Shou', [2,5]],
        ['Elbow', [3,6]],
        ['Wrist', [4,7]],
        ['Hip', [8,11]],
        ['Knee', [9,12]],
        ['Ankle', [10,13]],
    ]
    all_joints = []
    for i in joint_groups:
        all_joints += i[1]
    return joint_groups, all_joints


def h36m_compute_3d_pck(seq_err):
    pck_thresh = 150
    _, all_joints = h36m_joint_groups()
    pck_list = []
    for k, v in seq_err.items():
        err = np.stack(v, axis=0).astype(np.float32)
        pck_list.extend(list(np.float32(err[:, all_joints] < pck_thresh).mean(axis=1)))
    
    pck_3d = mean(pck_list)
    return pck_3d


def mpii_compute_3d_pck(seq_err, seq_mask):
    pck_thresh = 150
    _, all_joints = mpii_joint_groups()
    pck_list = []
    for k, v in seq_err.items():
        err = np.stack(v, axis=0).astype(np.float32)[:, all_joints]
        err_mask = np.stack(seq_mask[k], axis=0).astype(np.float32)[:, all_joints]
        masked_err = err * err_mask
        pck_3d = 1 - np.sum(masked_err > pck_thresh, axis=-1) / (np.sum(err_mask, axis=-1) + 1e-8)
        pck_list.extend(pck_3d.tolist())
    
    pck_3d = mean(pck_list)
    return pck_3d
