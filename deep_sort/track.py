# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 18:01
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : track.py
# @Software: PyCharm

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    pass


"""
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

"""

class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
            pass
        self._n_init = n_init
        self._max_age = max_age
        pass

    def to_tlwh(self):
        #(top left x, top left y, width, height)
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
        pass

    def to_tlbr(self):
        #(min x, miny, max x,max y)
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
        pass

    def predict(self,kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        pass

    def update(self,kf,detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
            pass
        pass

    def mark_missed(self):
        #Mark this track as missed (no association at the current time step).
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            pass
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
            pass
        pass

    def is_tentative(self):
        return self.state == TrackState.Tentative
        pass

    def is_confirmed(self):
        return self.state == TrackState.Confirmed
        pass

    def is_deleted(self):
        return self.state == TrackState.Deleted
        pass




    pass