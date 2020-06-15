# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 19:58
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : nn_matching.py
# @Software: PyCharm

import numpy as np

#欧式距离
def _euclidean_distance(a, b):
    #Compute pair-wise squared distance between points in `a` and `b`.
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a),len(b)))
        pass

    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:,None] +b2[None,:]
    r2 = np.clip(r2, 0.,float(np.inf))
    return r2
    pass

#余弦距离
#余弦距离=1-余弦相似度。
def _cosine_distance(a,b,data_is_normalize=False):
    if not data_is_normalize:
        a = np.asarray(a) / np.linalg.norm(a, axis=1,keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        pass

    return 1. - np.dot(a,b.T)
    pass

def _nn_euclidean_distance(x,y):
    #Helper function for nearest neighbor distance metric (Euclidean).
    distances = _euclidean_distance(x,y)
    return np.maximum(0.0,distances.min(axis=0))
    pass

def _nn_cosine_distance(x,y):
    distances = _cosine_distance(x,y)
    return distances.min(axis=0)
    pass

class NearestNeighborDistanceMetric(object):
    """ A nearest neighbor distance metric that, for each target, returns
        the closest distance to any sample that has been observed so far.
    """
    ## 对于每个目标，返回一个最近的距离

    def __init__(self,metric, matching_threshold, budget=None):
        # 默认matching_threshold = 0.2, budge = 100
        if metric == 'euclidean':
            # 使用最近邻欧氏距离
            self._metric = _nn_euclidean_distance
            pass
        elif metric == 'cosine':
            # 使用最近邻余弦距离
            self._metric = _nn_cosine_distance
            pass
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
            pass
        self.matching_threshold = matching_threshold
        # 在级联匹配的函数中调用
        self.budget = budget
        # budge 预算，控制feature的多少
        # samples是一个字典{id->feature list}
        self.samples = {}
        pass

    def partial_fit(self, features, targets, active_targets):
        #Update the distance metric with new data.
        #作用：部分拟合，用新的数据更新测量距离
        # 调用：在特征集更新模块部分调用，tracker.update()中
        for feature, target in zip(features,targets):
            self.samples.setdefault(target,[]).append(feature)
            # 对应目标下添加新的feature，更新feature集合
            # 目标id  :  feature list
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
                pass
            pass

        # 筛选激活的目标
        self.samples = {k:self.samples[k] for k in active_targets}
        pass

    def distance(self,features, targets):
        #Compute distance between features and targets.
        # 作用：比较feature和targets之间的距离，返回一个代价矩阵
        # 调用：在匹配阶段，将distance封装为gated_metric,
        # 进行外观信息(reid得到的深度特征) + 运动信息(马氏距离用于度量两个分布相似程度)
        cost_matrix = np.zeros((len(targets),len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i,:] = self._metric(self.samples[target],features)
            pass
        return cost_matrix
        pass

    pass

