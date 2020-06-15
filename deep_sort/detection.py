# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 10:03
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : detection.py
# @Software: PyCharm
import numpy as np
class Detection(object):
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh,dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        pass

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
                `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
        pass

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
                height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /=ret[3]
        return ret
        pass


    pass