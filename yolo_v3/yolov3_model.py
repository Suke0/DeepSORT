# -*- coding: utf-8 -*-
# @Time    : 2020/6/11 17:15
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : yolov3_model.py
# @Software: PyCharm
import tensorflow as tf
#from tensorflow.python.keras.models import load_model
import numpy as np
def Conv2D_BN_LeakyReLU(filters,kernel_size,strides,padding="same"):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,kernel_size,strides,padding=padding,use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1)
    ])
    pass

def Conv2D_BN_LeakyReLU_UpSampling2D(filters,kernel_size,strides,padding="same"):
    return tf.keras.Sequential([
        Conv2D_BN_LeakyReLU(filters,kernel_size,strides),
        tf.keras.layers.UpSampling2D(2)
    ])

def res_block(x,filters,n_blocks):
    x = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(x)
    x = Conv2D_BN_LeakyReLU(filters,3,2,padding='valid')(x)
    for i in range(n_blocks):
        x_ = tf.keras.Sequential([
            Conv2D_BN_LeakyReLU(filters//2,1,1),
            Conv2D_BN_LeakyReLU(filters,3,1)
        ])(x)
        x = tf.keras.layers.Add()([x,x_])
        pass
    return x
    pass

def darknet_body(x):
    x = Conv2D_BN_LeakyReLU(32,3,1)(x)
    x = res_block(x, 64, 1)
    x = res_block(x, 128, 2)
    x = res_block(x, 256, 8)
    x1 = x
    x = res_block(x, 512, 8)
    x2 = x
    x = res_block(x, 1024, 4)
    x3 = x
    return x1, x2, x3
    pass

def make_last_layers(x,n_filters,out_filters):
    x = tf.keras.Sequential([
        Conv2D_BN_LeakyReLU(n_filters,1,1),
        Conv2D_BN_LeakyReLU(n_filters * 2, 3, 1),
        Conv2D_BN_LeakyReLU(n_filters, 1, 1),
        Conv2D_BN_LeakyReLU(n_filters * 2, 3, 1),
        Conv2D_BN_LeakyReLU(n_filters, 1, 1)
    ])(x)
    x_ = tf.keras.Sequential([
        Conv2D_BN_LeakyReLU(n_filters * 2, 3, 1),
       tf.keras.layers.Conv2D(out_filters, 1, 1,padding='same')
    ])(x)
    return x, x_
    pass


def yolov3_model(n_classes,n_anchors):
    inputs = tf.keras.Input(shape=(416,416,3))
    x1, x2, x3 = darknet_body(inputs)
    x, y1 = make_last_layers(x3, 512, n_anchors * (n_classes + 5))

    x = Conv2D_BN_LeakyReLU_UpSampling2D(256,1,1)(x)
    x = tf.keras.layers.Concatenate()([x, x2])
    x,y2 = make_last_layers(x, 256, n_anchors * (n_classes + 5))

    x = Conv2D_BN_LeakyReLU_UpSampling2D(128,1,1)(x)
    x = tf.keras.layers.Concatenate()([x, x1])
    x, y3 = make_last_layers(x, 128, n_anchors * (n_classes + 5))
    return tf.keras.Model(inputs,[y1,y2,y3])
    pass

#@tf.function
def create_yolov3_model(inputs):
    model = yolov3_model(80,3)
    return model(inputs)
    pass

def tranfer_weight():

    imgs = np.random.randn(3, 416, 416, 3).astype(np.float32)
    input_tensor = tf.constant(imgs)
    # 4D tensor with shape: (samples, channels, rows, cols)
    # model = yolov3_model(80,3)

    model = tf.keras.models.load_model("./weight/yolo.h5")
    result = model(input_tensor)

    print(result[0].shape, result[1].shape, result[2].shape)
    vars = model.variables
    print(len(vars))
    # for v in model.variables:
    #     print(v.shape)
    # pass
    print(vars[345].shape)
    model_ = yolov3_model(80, 3)
    vars_ = model_.variables
    arr = np.array(vars[-20:])
    temp = arr[[2, 3, 4, 5, 14, 15, 0, 6, 7, 8, 9, 16, 17, 1, 10, 11, 12, 13, 18, 19]]
    vars[-20:] = list(temp)
    print(temp.shape)
    arr = []
    for i, var in enumerate(vars):
        arr.append(var.numpy())
        pass
    np.save('../yolo_weight.npy', arr, allow_pickle=True)
    # result = model(input_tensor)
    # print(result[0].shape, result[1].shape, result[2].shape)
    # result_ = model_(input_tensor)
    # print(result_[0].shape, result_[1].shape, result_[2].shape)
    # print(result[0].numpy()[0])
    # print("_____________________________")
    # print(result_[0].numpy()[0])

    #model_.save_weights("./weight/yolo_save_weights.h5")
    #model_.load_weights("./weight/yolo_save_weights.h5")
    #model_.save("./weight/yolo_save.h5")
    pass

def transfer_weight_npy(model):
    weights = np.load('./weight/yolo_weight.npy', allow_pickle=True)
    for v, v1 in zip(model.variables, weights):
        v.assign(v1)
        pass
    model.save("./weight/yolo_save.h5",)
    pass
if __name__ == "__main__":
    imgs = np.random.randn(3, 416, 416, 3).astype(np.float32)
    input_tensor = tf.constant(imgs)
    model_ = yolov3_model(80, 3)
    model_.load_weights("./weight/yolo_save_weight.h5")
    result_ = model_(input_tensor)
    print(result_[0].shape, result_[1].shape, result_[2].shape)
    model = tf.keras.models.load_model("./weight/yolo.h5")
    result = model(input_tensor)
    print(result[0].shape, result[1].shape, result[2].shape)
    #[[[ 5.39430201e-01  2.96730429e-01  8.41164768e-01 ... -4.76552057e+00

    print(result[0].numpy()[0])
    print("_________________________________")
    print(result_[0].numpy()[0])
    #transfer_weight_npy(model)
    pass