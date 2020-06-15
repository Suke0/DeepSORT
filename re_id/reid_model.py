# -*- coding: utf-8 -*-
# @Time    : 2020/6/9 16:30
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : reid_model.py
# @Software: PyCharm
import tensorflow as tf
def Conv2D_BN(filters,kernel_size,strides):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,kernel_size,strides=strides,padding='SAME',use_bias=False),
        tf.keras.layers.BatchNormalization(epsilon=0.001)
    ])
    pass

def Conv2D_BN_ELU(filters,kernel_size,strides):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,kernel_size,strides=strides,padding='SAME',use_bias=False),
        tf.keras.layers.BatchNormalization(epsilon=0.001),
        tf.keras.layers.ELU()
    ])
    pass

def res_block(inputs,filters,kernel_size,strides,is_first=False):
    x = inputs
    if not is_first:
        x = tf.keras.layers.BatchNormalization(epsilon=0.001)(x)
        x = tf.keras.layers.ELU()(x)
    if strides == 2:
        inputs = tf.keras.layers.Conv2D(filters,1,strides,padding="SAME",use_bias=False)(inputs)
        pass
    x = Conv2D_BN_ELU(filters, kernel_size, strides)(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, 1,padding="SAME")(x)
    x = tf.keras.layers.Add()([x,inputs])
    return x
    pass

# def l2norm(x, scale, trainable=True, scope="L2Normalization"):
#     n_channels = x.get_shape().as_list()[-1]
#     l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)
#     with tf.variable_scope(scope):
#         gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
#                                 initializer=tf.constant_initializer(scale),
#                                 trainable=trainable)
#         return l2_norm * gamma

def create_reid_model(inputs):
    model = reid_model()
    outputs = model(inputs)
    return outputs
    pass


def reid_model(): #inputs.shapes=[batch_size,128, 64, 3] RGB
    inputs = tf.keras.Input(shape=(128, 64, 3))
    x = Conv2D_BN_ELU(32, 3, 1)(inputs)
    x = Conv2D_BN_ELU(32, 3, 1)(x)
    x = tf.keras.layers.MaxPool2D(3,2,padding="VALID")(x)
    x = res_block(x, 32, 3, 1,True)
    x = res_block(x, 32, 3, 1)
    x = res_block(x, 64, 3, 2)
    x = res_block(x, 64, 3, 1)
    x = res_block(x, 128, 3, 2)
    x = res_block(x, 128, 3, 1)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128,use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=0.001)(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.BatchNormalization(epsilon=0.001)(x)
    outputs = tf.keras.layers.Lambda(lambda a : tf.math.l2_normalize(a, axis=1, epsilon=1e-8))(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs)
    pass

def transfor_weight(model):
    arr = []
    weights = np.load('./weight.npy', allow_pickle=True)
    for v in weights:
        print(v.shape)
        pass
    for v in model.variables:
        if not "gamma" in v.name:
            arr.append(v)
        pass
    for v, v1 in zip(arr, weights):
        v.assign(v1)
        pass

    model.save_weights("./reid_weight.h5")
    pass

if __name__ == '__main__':
    import numpy as np
    model = reid_model()
    # i = 0
    # for v in model.variables:
    #     if not "gamma" in v.name:
    #         print(v.name + "__" + str(v.shape) + "__" + str(i))
    #         i += 1
    #     pass
    # transfor_weight(model)
    model.load_weights("./reid_weight.h5")
    model.save("./reid_weight_save.h5")

    inputs = 128 * np.ones((1,128,64,3))

    inputs = tf.cast(inputs,dtype=tf.float32)

    x= model(inputs)
    print(x.numpy())

    pass
