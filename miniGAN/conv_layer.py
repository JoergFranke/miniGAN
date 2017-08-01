
import numpy as np
import tensorflow as tf
from math import ceil



def conv_layer(bottom, filter_shape, filter_depth, name, stride, padding='SAME'):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):

        pre_depth = bottom.get_shape()[3].value
        weights_shape = filter_shape + [pre_depth, filter_depth]

        weight = tf.get_variable(name+"_weight", weights_shape, initializer=tf.orthogonal_initializer(gain=1.0, seed=None),
                                 collections=['variables'])
        bias = tf.get_variable(name+"_bias", filter_depth, initializer=tf.constant_initializer(0.01),
                               collections=['variables'])

        conv = tf.nn.conv2d(bottom, weight, strides=strides, padding=padding)
        return tf.nn.relu(conv + bias)


def deconv_layer(bottom, filter_shape, filter_depth, name, stride, padding='SAME'):
    strides = [1, stride, stride, 1]

    with tf.variable_scope(name):

        pre_depth = bottom.get_shape()[3].value

        in_shape = tf.shape(bottom)

        h = in_shape[1] * stride
        w = in_shape[2] * stride
        output_shape = [in_shape[0], h, w, filter_depth]

        weights_shape = filter_shape + [filter_depth, pre_depth]

        width = filter_shape[0]
        heigh = filter_shape[1]
        f = ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(weights_shape)
        for i in range(weights_shape[2]):
            weights[:, :, i, i] = bilinear

        weight = tf.get_variable(name+"_up_weight", initializer=tf.constant_initializer(value=weights, dtype=tf.float32),
                                 shape=weights.shape, collections=['variables'])
        bias = tf.get_variable(name+"_bias", filter_depth, initializer=tf.constant_initializer(0.01),
                               collections=['variables'])

        deconv = tf.nn.conv2d_transpose(bottom, weight, output_shape, strides, padding=padding, data_format='NHWC',
                                        name="deconv")

        return tf.nn.relu(deconv + bias)



def maxp_layer(bottom, name, pooling=2):
    return tf.nn.max_pool(bottom, ksize=[1, pooling, pooling, 1],
                          strides=[1, pooling, pooling, 1], padding='SAME', name=name)


def sigmoid_layer(bottom, name, weighted=False, num_classes=None, pre_depth=None):
    with tf.variable_scope(name):


        in_shape = tf.shape(bottom)


        if weighted:
            weights_shape = [pre_depth , num_classes]
            weight = tf.get_variable(name+"_weight", initializer=tf.truncated_normal(weights_shape, stddev=0.01),
                                     collections=['variables'])
            bias = tf.get_variable(name+"_bias", num_classes, initializer=tf.constant_initializer(0.01),
                                   collections=['variables'])

            flat_bottom = tf.reshape(bottom, (-1, in_shape[3]))
            flat_bottom = tf.matmul(flat_bottom, weight) + bias
        else:
            flat_bottom = tf.reshape(bottom, (-1, in_shape[3]))

        flat_pred = tf.sigmoid(flat_bottom)

        prediction = tf.reshape(flat_pred, (in_shape[0],in_shape[1],in_shape[2],num_classes))

    return prediction, flat_pred


def conv_norm_acti_drop_layer(bottom, filter_shape, filter_depth, name, stride, padding='SAME', keep_prob=0.8, norm=True, pre_depth=None, depthwise=False):
    strides = [1, stride, stride, 1]
    with tf.variable_scope('conv_' + name):
        if pre_depth== None:
            pre_depth = bottom.get_shape()[3].value
        weights_shape = filter_shape + [pre_depth, filter_depth]
        # weight = tf.get_variable("weight", initializer=tf.truncated_normal(weights_shape, stddev=0.01),
        #                          collections=['variables'])

        minval = -tf.sqrt(2 / (filter_shape[0] * filter_shape[1] * pre_depth +  filter_depth))
        maxval =  tf.sqrt(2 / (filter_shape[0] * filter_shape[1] * pre_depth +  filter_depth))
        weight = tf.get_variable("weight",weights_shape, initializer=tf.random_uniform_initializer(minval, maxval, seed=None),
                                 collections=['variables'])

        # weight = tf.get_variable("weight", weights_shape,
        #                          initializer=tf.orthogonal_initializer(gain=1.0, seed=None),
        #                          collections=['variables'])

        if depthwise:
            bias = tf.get_variable("bias", filter_depth * pre_depth, initializer=tf.constant_initializer(0.01),
                                   collections=['variables'])
            conv = tf.nn.depthwise_conv2d(bottom, weight, strides, padding, name=None)
        else:
            if padding=='NEIGH':
                bias = tf.get_variable("bias", filter_depth, initializer=tf.constant_initializer(0.01),
                                       collections=['variables'])
                pad_value = int((filter_shape[0]-1)/2)
                print("NEIGH {}".format(pad_value))
                padd_input = tf.pad(bottom, [[0,0],[pad_value,pad_value],[pad_value,pad_value],[0,0]], mode='CONSTANT')
                conv = tf.nn.conv2d(padd_input, weight, strides=[1,1,1,1], padding='VALID')

            else:
                bias = tf.get_variable("bias", filter_depth, initializer=tf.constant_initializer(0.01),
                                       collections=['variables'])
                conv = tf.nn.conv2d(bottom, weight, strides=strides, padding=padding)



    if norm:
        with tf.variable_scope('norm_' + name) as ln:
            #normalized = tf.contrib.layers.layer_norm(conv, center=True, scale=False, scope=ln)
            normalized = tf.contrib.layers.layer_norm(conv, center=True, scale=True, scope=ln)
    else:
        normalized = conv


    with tf.variable_scope('acti_' + name):
        activated = tf.nn.elu(normalized + bias)

    if keep_prob == False:
        dropouted = activated
    else:
        with tf.variable_scope('drop_' + name):
            dropouted = tf.nn.dropout(activated, keep_prob)

    return dropouted