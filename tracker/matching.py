import numpy as np
from utils.utils import polygon_iou
import cv2
from scipy.optimize import linear_sum_assignment

def _indices_to_matches(cost_matrix, indices, thresh):
    indices_=[]
    for i in zip(indices[0], indices[1]):
        indices_.append(i)
    indices=np.array(indices_)
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)
    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b

def linear_assignment(cost_matrix, thresh):
    """
    Simple linear assignment
    :type cost_matrix: np.ndarray
    :type thresh: float
    :return: matches, unmatched_a, unmatched_b
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    indices = linear_sum_assignment(cost_matrix)

    return _indices_to_matches(cost_matrix, indices, thresh)

def poly_distance(atracks, btracks):
    
    apts = [track.pt for track in atracks]
    bpts = [track.pt for track in btracks]
    apts = np.ascontiguousarray(apts, dtype=np.float)
    bpts = np.ascontiguousarray(bpts, dtype=np.float)

    if apts.size==0 or bpts.size==0:
        _ious = np.zeros((len(apts), len(bpts)), dtype=np.float)
    else:
        _ious = polygon_iou(apts, bpts)
    cost_matrix = 1 - _ious
    return cost_matrix


def rec_dis(apts, bpts):
    dis = np.zeros((len(apts), len(bpts)), dtype=np.float)
    if dis.size == 0:
        return dis
    apts = np.ascontiguousarray(apts, dtype=np.float)
    bpts = np.ascontiguousarray(bpts, dtype=np.float)
    apts = np.expand_dims(apts, 1)
    apts = np.tile(apts, (1, bpts.shape[0], 1))
    bpts = np.expand_dims(bpts, 0)
    bpts = np.tile(bpts, (apts.shape[0], 1, 1))

    dis = abs(apts - bpts)
    dis[:, :, :4] /= 10
    dis[:, :, 5][dis[:, :, 5]>45] = 90 - dis[:, :, 5][dis[:, :, 5]>45]
    dis[:, :, 5] /= 10
    dis = 0.3*(dis[:, :, :4].sum(axis=-1))/(4*dis[:,:,6]) + 0.7*dis[:,:,4:6].sum(axis=-1)/2
    return dis

def shape_distance(atracks, btracks):
    
    apts = []
    for track in atracks:
        ct,wh,th = cv2.minAreaRect(track.pt.reshape(4,2))
        if wh[0]<wh[1]:
            h,w=wh
        else:
            w,h=wh
        apts.append(list(ct)+[w,h,w/(h+1e-5),th,track.frame_id])

    bpts = []
    for track in btracks:
        ct,wh,th = cv2.minAreaRect(track.pt.reshape(4,2))
        if wh[0]<wh[1]:
            h,w=wh
        else:
            w,h=wh
        bpts.append(list(ct)+[w,h,w/(h+1e-5),th,track.cur_frame])

    dis = rec_dis(apts, bpts)
    cost_matrix = dis
    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[STrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        print('no STrack', len(tracks), len(detections))
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)

    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.sum(np.power((np.repeat(track.curr_feat.reshape(1,-1), len(det_features), 0) - det_features),2), 1)
    return cost_matrix
