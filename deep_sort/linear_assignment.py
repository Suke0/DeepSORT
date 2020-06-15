# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 19:59
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : linear_assignment.py
# @Software: PyCharm

import numpy as np
from . import kalman_filter
from sklearn.utils.linear_assignment_ import linear_assignment
INFTY_COST = 1e+5

def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None,detection_indices=None):
    if track_indices is None:
        track_indices = np.arange((len(tracks)))
        pass

    if detection_indices is None:
        detection_indices = np.arange(len(detections))
        pass

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices
        pass

    cost_matrix = distance_metric(tracks,detections,track_indices,detection_indices)
    cost_matrix[cost_matrix>max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [],[],[]
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:,1]:
            unmatched_detections.append(detection_idx)
            pass
        pass

    for row, track_idx in enumerate(track_indices):
        if col not in indices[:,0]:
            unmatched_tracks.append(track_idx)
            pass
        pass

    for row,col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row,col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
            pass
        else:
            matches.append((track_idx,detection_idx))
            pass
        pass
    return matches, unmatched_tracks, unmatched_detections
    pass

def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
        pass
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
        pass

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
            pass
        track_indices_1 = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_1) == 0:
            continue
            pass

        matches_1, _, unmatched_detections = min_cost_matching(distance_metric,max_distance,tracks,detections,track_indices_1,unmatched_detections)
        matches += matches_1
        pass

    unmatched_tracks = list(set(track_indices) - set(k for k,_ in matches))
    return matches, unmatched_tracks,unmatched_detections
    pass

def gate_cost_matrix(kf,cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=INFTY_COST,only_position=False):
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_inx in enumerate(track_indices):
        track = tracks[track_inx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row,gating_distance>gating_threshold] = gated_cost
        pass
    return cost_matrix
    pass