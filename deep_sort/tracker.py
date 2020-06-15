# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 18:01
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : tracker.py
# @Software: PyCharm
import numpy as np

from deep_sort import linear_assignment, iou_matching
from . import kalman_filter
from .track import Track

"""
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

"""

class Tracker:
    # 是一个多目标tracker，保存了很多个track轨迹
    # 负责调用卡尔曼滤波来预测track的新状态+进行匹配工作+初始化第一帧
    # Tracker调用update或predict的时候，其中的每个track也会各自调用自己的update或predict
    def __init__(self, metric, max_iou_distance = 0.7, max_age = 30, n_init = 3 ): # 调用的时候，后边的参数全部是默认的
        # metric是一个类，用于计算距离(余弦距离或马氏距离) NearestNeighborDistanceMetric
        self.metric = metric
        # 最大iou，iou匹配的时候使用
        self.max_iou_distance = max_iou_distance
        # 直接指定级联匹配的cascade_depth参数
        self.max_age = max_age
        # n_init代表需要n_init次数的update才会将track状态设置为confirmed
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()  # 卡尔曼滤波器
        self.tracks = []  # 保存一系列轨迹
        self._next_id = 1  # 下一个分配的轨迹id
        pass

    def _initiate_track(self,detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age,detection.feature))
        self._next_id += 1
        pass

    def predict(self):
        # Propagate track state distributions one time step forward.
        # This function should be called once every time step, before `update`.
        # 遍历每个track都进行一次预测
        for track in self.tracks:
            track.predict(self.kf)
            pass
        pass

    def update(self,detections):
        #Perform measurement update and track management.
        #detections : List[deep_sort.detection.Detection] A list of detections at the current time step.
        # 进行测量的更新和轨迹管理

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        #Update track set
        # 1. 针对匹配上的结果
        for track_idx, detection_idx in matches:
            # track更新对应的detection
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
            pass

        # 2. 针对未匹配的tracker,调用mark_missed标记
        # track失配，若待定则删除，若update时间很久也删除
        # max age是一个存活期限，默认为70帧
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            pass

        # 3. 针对未匹配的detection， detection失配，进行初始化
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
            pass

        # 得到最新的tracks列表，保存的是标记为confirmed和Tentative的track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        #Update distance netric
        # 获取所有confirmed状态的track id
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        featrues, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
                pass

            # 将tracks列表拼接到features列表
            featrues += track.features
            # 获取每个feature对应的track id
            targets += [track.track_id for _ in track.features]
            track.features = []
            pass

        # 距离度量中的 特征集更新
        self.metric.partial_fit(np.asarray(featrues),np.asarray(targets),active_targets)
        pass

    def _match(self,detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
            return cost_matrix
            pass

        confirmed_tracks = [ i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [ i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatchd_detections = linear_assignment.matching_cascade(gated_metric,self.metric.matching_threshold, self.max_age,self.tracks,detections,confirmed_tracks)

        iou_track_candidates = unconfirmed_tracks +[ k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [ k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatchd_detections = linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance,self.tracks,detections,iou_track_candidates,unmatchd_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatchd_detections
        pass

    def _initiate_track(self,detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))
        self._next_id += 1
        pass
    pass