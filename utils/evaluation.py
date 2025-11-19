# Modified from agora_evaluation, OSX
from itertools import product
import numpy as np
from .transforms import rigid_align, pelvis_align, root_align

def vectorize_distance(a, b):
    """
    Calculate euclid distance on each row of a and b
    :param a: Nx... np.array
    :param b: Mx... np.array
    :return: MxN np.array representing correspond distance
    """
    N = a.shape[0]
    a = a.reshape ( N, -1 )
    M = b.shape[0]
    b = b.reshape ( M, -1 )
    a2 = np.tile ( np.sum ( a ** 2, axis=1 ).reshape ( -1, 1 ), (1, M) )
    b2 = np.tile ( np.sum ( b ** 2, axis=1 ), (N, 1) )
    dist = a2 + b2 - 2 * (a @ b.T)
    return np.sqrt(dist)



# for agora evaluation
def select_and_align(smpl_joints, smpl_verts, body_verts_ind):
    joints = smpl_joints[:24, :]
    verts = smpl_verts[body_verts_ind, :]
    assert len(verts.shape) == 2
    verts = pelvis_align(joints, verts)
    joints = pelvis_align(joints)
    return joints, verts


def l2_error(j1, j2):
    return np.linalg.norm(j1 - j2, 2)


def compute_prf1(count, miss, num_fp):
    if count == 0:
        return 0, 0, 0
    all_tp = count - miss
    all_fp = num_fp
    all_fn = miss
    all_f1_score = round(all_tp / (all_tp + 0.5 * (all_fp + all_fn)), 2)
    all_recall = round(all_tp / (all_tp + all_fn), 2)
    all_precision = round(all_tp / (all_tp + all_fp), 2)
    return all_precision, all_recall, all_f1_score

# mean per-vertex position error
def cal_3d_position_error(pred, gt):
    """mve,pa_mve,mpjpe,pa_mpjpe
    """
    assert len(pred.shape) == 2 and pred.shape == gt.shape

    error = (np.sqrt(np.sum((pred - gt) ** 2, 1)).mean() * 1000)
    pred_align = rigid_align(pred,gt)
    pa_error = (np.sqrt(np.sum((pred_align - gt) ** 2, 1)).mean() * 1000)
    return error, pa_error


def match_2d_greedy(
        pred_kps,
        gtkp,
        iou_thresh=0.1,
        valid=None):
    '''
    matches groundtruth keypoints to the detection by considering all possible matchings.
    :return: best possible matching, a list of tuples, where each tuple corresponds to one match of pred_person.to gt_person.
            the order within one tuple is as follows (idx_pred_kps, idx_gt_kps)
    '''

    nkps = 24

    predList = np.arange(len(pred_kps))
    gtList = np.arange(len(gtkp))
    # get all pairs of elements in pred_kps, gtkp
    # all combinations of 2 elements from l1 and l2
    combs = list(product(predList, gtList))

    errors_per_pair = {}
    errors_per_pair_list = []
    for comb in combs:
        errors_per_pair[str(comb)] = l2_error(
            pred_kps[comb[0]][:nkps, :2], gtkp[comb[1]][:nkps, :2])
        errors_per_pair_list.append(errors_per_pair[str(comb)])

    gtAssigned = np.zeros((len(gtkp),), dtype=bool)
    opAssigned = np.zeros((len(pred_kps),), dtype=bool)
    errors_per_pair_list = np.array(errors_per_pair_list)

    bestMatch = []
    excludedGtBecauseInvalid = []
    falsePositiveCounter = 0
    while np.sum(gtAssigned) < len(gtAssigned) and np.sum(
            opAssigned) + falsePositiveCounter < len(pred_kps):
        found = False
        falsePositive = False
        while not(found):
            # if sum(np.inf == errors_per_pair_list) == len(
            #         errors_per_pair_list):
            #     logging.fatal('something went wrong here')

            minIdx = np.argmin(errors_per_pair_list)
            minComb = combs[minIdx]
            # compute IOU
            iou = get_bbx_overlap(
                pred_kps[minComb[0]], gtkp[minComb[1]])
            # if neither prediction nor ground truth has been matched before and iou
            # is larger than threshold
            if not(opAssigned[minComb[0]]) and not(
                    gtAssigned[minComb[1]]) and iou >= iou_thresh:
                found = True
                errors_per_pair_list[minIdx] = np.inf
            else:
                errors_per_pair_list[minIdx] = np.inf
                # if errors_per_pair_list[minIdx] >
                # matching_threshold*headBboxs[combs[minIdx][1]]:
                if iou < iou_thresh:
                    found = True
                    falsePositive = True
                    falsePositiveCounter += 1

        # if ground truth of combination is valid keep the match, else exclude
        # gt from matching
        if not(valid is None):
            if valid[minComb[1]]:
                if not falsePositive:
                    bestMatch.append(minComb)
                    opAssigned[minComb[0]] = True
                    gtAssigned[minComb[1]] = True
            else:
                gtAssigned[minComb[1]] = True
                excludedGtBecauseInvalid.append(minComb[1])

        elif not falsePositive:
            # same as above but without checking for valid
            bestMatch.append(minComb)
            opAssigned[minComb[0]] = True
            gtAssigned[minComb[1]] = True

    # add false positives and false negatives to the matching
    # find which elements have been successfully assigned
    opAssigned = []
    gtAssigned = []
    for pair in bestMatch:
        opAssigned.append(pair[0])
        gtAssigned.append(pair[1])
    opAssigned.sort()
    gtAssigned.sort()

    # handle false positives
    opIds = np.arange(len(pred_kps))
    # returns values of oIds that are not in opAssigned
    notAssignedIds = np.setdiff1d(opIds, opAssigned)
    for notAssignedId in notAssignedIds:
        bestMatch.append((notAssignedId, 'falsePositive'))
    gtIds = np.arange(len(gtList))
    # returns values of gtIds that are not in gtAssigned
    notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned)

    # handle false negatives/misses
    for notAssignedIdGt in notAssignedIdsGt:
        if not(valid is None):  # if using the new matching
            if valid[notAssignedIdGt]:
                bestMatch.append(('miss', notAssignedIdGt))
            else:
                excludedGtBecauseInvalid.append(notAssignedIdGt)
        else:
            bestMatch.append(('miss', notAssignedIdGt))

    # handle invalid ground truth
    for invalidGt in excludedGtBecauseInvalid:
        bestMatch.append(('invalid', invalidGt))

    return bestMatch  # tuples are (idx_pred_kps, idx_gt_kps)


def get_bbx_overlap(p1, p2):

    min_p1 = np.min(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p1 = np.max(p1, axis=0)
    max_p2 = np.max(p2, axis=0)

    bb1 = {}
    bb2 = {}

    bb1['x1'] = min_p1[0]
    bb1['x2'] = max_p1[0]
    bb1['y1'] = min_p1[1]
    bb1['y2'] = max_p1[1]
    bb2['x1'] = min_p2[0]
    bb2['x2'] = max_p2[0]
    bb2['y1'] = min_p2[1]
    bb2['y2'] = max_p2[1]

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou


def get_matching_dict(matching):
    matchDict = {}
    falsePositive_count = 0
    for match in matching:
        if not (match[1] == 'falsePositive') or match[0] == 'invalid':
            # tuple order (idx_openpose_pred, idx_gt_kps)
            matchDict[str(match[1])] = match[0]
        elif (match[1] == 'falsePositive'):
            falsePositive_count += 1
        else:
            continue  # simply ignore invalid ground truths
    return matchDict, falsePositive_count


def calc_MPVPE(pred_verts, gt_verts):
    """Also calculate PA-MPVPE
        -pred_verts: shape (bs,num_person,num_verts,3), root ralative
        -gt_verts: same format as pred_verts
    """
    assert len(pred_verts.shape) == 3 and pred_verts.shape == gt_verts.shape

    n,_,_ = pred_verts.shape

    res=[]
    pa_res=[]

    for idx in range(n):
        pred,gt = pred_verts[idx],gt_verts[idx]
        res.append(np.sqrt(np.sum((pred - gt) ** 2, 1)).mean() * 1000)
        pred_align = rigid_align(pred,gt)
        pa_res.append(np.sqrt(np.sum((pred_align - gt) ** 2, 1)).mean() * 1000)
    return res,pa_res