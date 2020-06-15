# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 10:12
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : kalman_filter.py
# @Software: PyCharm

import numpy as np
import scipy

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
                1: 3.8415,
                2: 5.9915,
                3: 7.8147,
                4: 9.4877,
                5: 11.070,
                6: 12.592,
                7: 14.067,
                8: 15.507,
                9: 16.919
                }
# 卡尔曼滤波首先根据当前帧(time=t)的状态进行预测，得到预测下一帧的状态(time=t+1)
# 得到测量结果，在Deep SORT中对应的测量就是Detection，即目标检测器提供的检测框。
# 将预测结果和测量结果进行更新。
class KalmanFilter(object):
    """
       A simple Kalman filter for tracking bounding boxes in image space.

       The 8-dimensional state space

           x, y, a, h, vx, vy, va, vh

       contains the bounding box center position (x, y), aspect ratio a, height h,
       and their respective velocities.

       Object motion follows a constant velocity model. The bounding box location
       (x, y, a, h) is taken as direct observation of the state space (linear
       observation model).

    """
    def __init__(self):
        ndim,dt = 4, 1.
        #Create Kalman filter model matrices

        self.A = np.eye(2*ndim, 2*ndim) # A矩阵
        for i in range(ndim):
            self.A[i,ndim+i] = dt
            pass
        self.H = np.eye(ndim, 2*ndim) # H矩阵
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1./20
        self._std_weight_velocity = 1./160
        pass

    def initiate(self,measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos,mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]

        covariance = np.diag(np.square(std))
        return mean, covariance  #X0和P0
        pass

    def predict(self,mean, covariance):
        # 相当于得到t时刻估计值
        # Q 预测过程中噪声协方差
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        # np.r_ 按列连接两个矩阵
        # 初始化噪声矩阵Q
        Q = np.diag(np.square(np.r_[std_pos,std_vel]))

        # x' = Ax
        mean = np.dot(self.A,mean)
        # P' = APA^T + Q
        covariance = np.linalg.multi_dot((self.A, covariance, self.A.T)) + Q
        return mean, covariance
        pass

    def project(self,mean, covariance):
        # R 测量过程中噪声的协方差
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]

        # 初始化噪声矩阵R
        R = np.diag(np.square(std))

        # 将均值向量映射到检测空间，即Hx'
        mean = np.dot(self.H, mean)

        # 将协方差矩阵映射到检测空间，即HP'H^T
        covariance = np.linalg.multi_dot((self.H,covariance,self.H.T)) + R
        return mean, covariance
        pass

    def update(self,mean,covariance,measurement):
        # 通过估计值和观测值估计最新结果

        # 将均值和协方差映射到检测空间，得到 Hx' 和 S
        #covariance 为 P'
        projected_mean, projected_cov = self.project(mean, covariance)

        # 矩阵分解 projected_cov为HP'H^T+R
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov,lower=True,check_finite=False)

        # 计算卡尔曼增益K
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                      np.dot(covariance, self.H.T).T,check_finite=False).T

        # z - Hx'
        innovation = measurement - projected_mean

        # x = x' + Ky
        new_mean = mean + np.dot(innovation, kalman_gain.T)

        # P = (I - KH)P'
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
        pass

    # z是Detection的mean，不包含变化值，状态为[cx, cy, a, h]。H是测量矩阵，将Track的均值向量映射到检测空间。计算的y是Detection和Track的均值误差。
    #
    # R是目标检测器的噪声矩阵，是一个4x4的对角矩阵。对角线上的值分别为中心点两个坐标以及宽高的噪声。
    #
    # 计算的是卡尔曼增益，是作用于衡量估计误差的权重。
    #
    # 更新后的均值向量x。
    #
    # 更新后的协方差矩阵。

    def gating_distance(self,mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.
            A suitable distance threshold can be obtained from `chi2inv95`. If
            `only_position` is False, the chi-square distribution has 4 degrees of
            freedom, otherwise 2.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2,:2]
            measurements = measurements[:,:2]
            pass

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor,d.T,lower = True, check_finite = False, overwrite_b =True)
        squared_maha = np.sum(z*z,axis=0)
        return squared_maha
        pass

    pass
