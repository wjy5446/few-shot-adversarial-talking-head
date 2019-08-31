import numpy as np

import tensorflow as tf
import tensorflow.contrib as tf_contrib

########################################
# Basic layer
########################################

def fc(x, channel, bias=False, sn=False, scope='fc'):
    '''
    make fully connect layer with xavier initializer and spectral normalization
    '''

    with tf.variable_scope(scope):
        # x (B, L, C)
        x = flatten(x) # x(B, LC)

        weight_shape = [x.get_shape().as_list()[-1], channel] # (LC, channel)

        if sn:
            w = tf.get_variable('weight', weight_shape, tf.float32,
                                initializer=tf_contrib.layers.xavier_initializer())

            if bias:
                b = tf.get_variable('bias', [weight_shape[-1]], tf.float32,
                                      initializer=tf.constant_initializer(0.0))
                x = tf.matmul(x, spectral_norm(w)) + b
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=channel, kernel_initializer=tf_contrib.layers.xavier_initializer(), use_bias=bias)

        return x

def conv(x, channels, kernel, stride, pad, pad_type='zero', use_bias=True, sn=False, scope='conv'):
    """
    make convolution layer with xvaier_initializer() and spectral_normalization
    """
    
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode = 'REFLECT')

        if sn:
            w = tf.get_variable('kernel', shape=[kernel, kernel, x.get_shape()[-1], channels],
                                initializer=tf_contrib.layers.xavier_initializer())

            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')

            if use_bias:
                b = tf.get_variable('bias', [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, b)
        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=tf_contrib.layers.xavier_initializer(),
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, pad_type='same', use_bias=True, sn=False, scope='deconv'):
    """
    make deconvolution layer with xvaier_initializer() and spectral_normalization
    """
    
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        
        if pad_type == 'same':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
                            x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn:
            w = tf.get_variable('kernel', shape=[kernel, kernel, channels, x.get_shape()[-1]],
                                initializer=tf_contrib.layers.xavier_initializer())

            x = tf.nn.conv2d_transpose(input=x, filter=spectral_norm(w), output_shape=output_shape,
                             strides=[1, stride, stride, 1], padding=padding)

            if use_bias:
                b = tf.get_variable('bias', [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, b)
        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                        kernel_size=kernel, kernel_initializer=tf_contrib.layers.xavier_initializer(),
                                        stride=stride, use_bias=bias)

        return x

########################################
# Activation fucntion
########################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)

def tanh(x):
    return tf.nn.tanh(x)

########################################
# Normalization fucntion 
########################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
                                        center=True, scale=True,
                                        updates_collections=None,
                                        is_training=is_training, scope=scope)

def instance_norm(x, is_training=True, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x, center=True, scale=True,
                                           trainable=is_training, scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()

    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


########################################
# function
########################################

def avg_pooling(x):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

def max_pooling(x):
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')

def global_avg_pooling(x):
    return tf.reduce_mean(x, axis=[1, 2])

def global_sum_pooling(x):
    return tf.reduce_sum(x, axis=[1, 2])

def up_sampling_nn(x, scale_factor=2):
    # x (B, H, W, C)
    _, h, w, _ = x.get_shape().as_list()
    size_new = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size_new)    

########################################
# function
########################################

def flatten(x):
    return tf.layers.flatten(x)

########################################
# misc
########################################



if __name__ == "__main__":
    a = [1, 2, 3]
    print(a[:-1])
    pass


