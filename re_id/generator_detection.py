# -*- coding: utf-8 -*-
# @Time    : 2020/6/12 9:43
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : generator_detection.py
# @Software: PyCharm
import cv2
import os
import numpy as np
import tensorflow as tf

from re_id.reid_model import reid_model


def run_in_batches(f,data_dict,out,batch_size):
    data_len = len(out)
    num_batches = int(data_len/batch_size)

    s, e =0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i+1) * batch_size
        batch_data_dict = {k:v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
        pass
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)
        pass
    pass

def extract_image_patch(image,bbox,patch_shape):
    """Extract image patch from bounding box.
    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1])/patch_shape[0]
        new_width = target_aspect *bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width
        pass
    # convert to top left, bottom right
    bbox[2:]+=bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0,bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1])-1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
        pass

    sx,sy,ex,ey = bbox
    image = image[sy:ey,sx:ex]
    image = cv2.resize(image,tuple(patch_shape[::-1]))
    return image
    pass

#@tf.function
def box_encoder(model_filename):
    model = reid_model()
    model.load_weights(model_filename)
    def encoder(image, boxes, batch_size=32):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, (128, 64))
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0., 255., image.shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches, dtype=np.float32)

        out = np.zeros((len(image_patches), 128), np.float32)
        data_len = len(out)
        num_batches = int(data_len / batch_size)

        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            out[s:e] = model(image_patches[s:e])
            pass
        if e < len(out):
            out[e:] = model(image_patches[e:])
            pass
        return out
        pass
    return encoder
    pass


def generate_detections(box_encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.
    """
    if detection_dir is None:
        detection_dir = mot_dir
        pass

    for sequence in os.listdir(mot_dir):
        sequence_dir = os.path.join(mot_dir, sequence)
        image_dir = os.path.join(sequence_dir,"img1")
        image_filenames = {int(os.path.splitext(f)[0]): os.path.join(image_dir,f) for f in os.listdir(image_dir)}

        detection_file = os.path.join(detection_dir, sequence,"det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detection_out = []

        frame_indices = detections_in[:,0].astype(np.int)
        min_frame_idx = frame_indices.min()
        max_frame_idx = frame_indices.max()

        for frame_idx in range(min_frame_idx,max_frame_idx+1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                continue
                pass

            bgr_iamge = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = box_encoder(bgr_iamge,rows[:,2:6].copy())
            detection_out += [np.r_[(row,feature)] for row, feature in zip(rows,features)]
            pass
        output_filename = os.path.join(output_dir,"%s.npy"%sequence)
        np.save(output_filename,np.asarray(detection_out),allow_pickle=False)
        pass
    pass