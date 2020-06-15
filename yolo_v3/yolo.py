# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 14:36
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : main.py
# @Software: PyCharm
import colorsys
import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

# import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/det_t1_video_00315_test.avi")
# ap.add_argument("-c", "--class",help="name of class", default = "person")
# args = vars(ap.parse_args())
from yolo_v3.yolov3_model import yolov3_model


class YOLOV3(object):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir+'/weight/yolo_save_weight.h5')
        anchors_path = os.path.join(current_dir+'/weight/yolo_anchors.txt')
        classes_path = os.path.join(current_dir+'/weight/coco_classes.txt')
        self.model_path = model_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path

        #具体参数可实验后进行调整
        self.score = 0.6 #0.8
        self.iou = 0.6
        self.model_image_size = (416,416)
        
        self.class_names = self._get_class()
        self.num_classes = len(self.class_names)
        self.anchors = self._get_anchors()
        self.initial()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def initial(self):
        model_path = os.path.expanduser(self.model_path)

        self.yolo_model = yolov3_model(len(self.class_names),3)
        self.yolo_model.load_weights(model_path)
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        pass

    def output_handle(self,inputs_tensor,image_shape,max_boxes=20,score_threshold=0.6,iou_threshold=0.5):
        """Evaluate YOLO model on given input and return filtered boxes."""
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        yolo_outputs = self.yolo_model(inputs_tensor)
        input_shape = inputs_tensor.shape[1:3]
        boxes = []
        box_scores = []
        for i in range(3):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[i],
                                                        self.anchors[anchor_mask[i]], self.num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes =tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(self.num_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)

        return boxes_, scores_, classes_
        pass

    #@tf.function
    def detect_image(self, image):
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.output_handle(image_data,image.size)

        return_boxs = []
        return_class_name = []
        person_counter = 0
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != 'person':
               continue
            person_counter += 1
            box = out_boxes[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3]-box[1])
            h = int(box[2]-box[0])
            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0
            return_boxs.append([x,y,w,h])
            return_class_name.append([predicted_class])
        #cv2.putText(image, str(self.class_names[c]),(int(box[0]), int(box[1] -50)),0, 5e-3 * 150, (0,255,0),2)
        #print("Found person: ",person_counter)
        return return_boxs,return_class_name
        pass

    pass

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128,128,128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    return boxed_image
    pass

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores
    pass

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape, image_shape =np.array(list(input_shape)), np.array([image_shape[1],image_shape[0]])

    new_shape = np.round(image_shape * np.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ],axis=-1)

    # Scale boxes back to original image shape.
    img_tensor = tf.concat([image_shape, image_shape],axis=-1)
    img_tensor = tf.cast(tf.expand_dims(img_tensor,axis=0),tf.float32)
    boxes *= img_tensor
    return boxes
    pass
def yolo_head(feats, anchors, num_classes, input_shape):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])

    grid_shape = np.shape(feats)[1:3] # height, width
    grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y],axis=-1)

    feats = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_xy = tf.sigmoid(feats[..., :2])
    box_wh = tf.exp(feats[..., 2:4])
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (box_xy + grid) / tf.cast(grid_shape[::-1],tf.float32)
    box_wh = box_wh * anchors_tensor / tf.cast(input_shape[::-1],tf.float32)

    return box_xy, box_wh, box_confidence, box_class_probs
    pass

if __name__ == "__main__":
    #model = YOLOV3()
    print(tf.test.gpu_device_name())
    pass