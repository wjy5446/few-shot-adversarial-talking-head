from layers import * 

import tensorflow as tf


def ResBlock_adaIN(x, psi, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        # x (B, H, W, C)
        # psi (B, psi_C)
        
        x_init = x
        with tf.variable_scope('res0'):
            out = adaIN(x, psi)
            out = relu(out)
            out = conv(x, channels, 3, 1, 1, pad_type='reflect', use_bias=use_bias, sn=sn)
        
        with tf.variable_scope('res1'):
            out = adaIN(out, psi)
            out = relu(out)
            out = conv(out, channels, 3, 1, 1, pad_type='reflect', use_bias=use_bias, sn=sn)
       
        print(out)
        return out + x_init

def ResBlockDown_g(x, channels, use_bias=True, sn=False, is_training=True, scope='resblock_down_g'): 
    with tf.variable_scope(scope):
        # x (B, H, W, C)

        with tf.variable_scope('res0_r'):
            out_r = instance_norm(x, is_training=is_training)
            out_r = lrelu(out_r, 0.2)
            out_r = conv(out_r, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        with tf.variable_scope('res1_r'):
            out_r = instance_norm(out_r, is_training=is_training)
            out_r = lrelu(out_r, 0.2)
            out_r = conv(out_r, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            out_r = avg_pooling(out_r)

        with tf.variable_scope('res0_l'):
            out_l = conv(x, channels, kernel=1, stride=1, pad=0, pad_type='zero', use_bias=use_bias, sn=sn)
            out_l = avg_pooling(out_l)
       
        print(out_r)
        print(out_l)
        return out_r + out_l

def ResBlockDown_d(x, channels, use_bias=True, sn=False, scope='resblock_down_d'):
    with tf.variable_scope(scope):
        # x (B, H, W, C)

        with tf.variable_scope('res0_r'):
            out_r = lrelu(x, 0.2)
            out_r = conv(out_r, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        with tf.variable_scope('res1_r'):
            out_r = lrelu(out_r, 0.2)
            out_r = conv(out_r, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            out_r = avg_pooling(out_r)

        with tf.variable_scope('res0_l'):
            out_l = conv(x, channels, kernel=1, stride=1, pad=0, pad_type='zero', use_bias=use_bias, sn=sn)
            out_l = avg_pooling(out_l)
       
        print(out_r)
        print(out_l)
        return out_r + out_l

def ResBlockUp_adaIN(x, psi, channels, use_bias=True, sn=False, scope='resblock_up'): 
    with tf.variable_scope(scope):
        # x (B, H, W, C)
        # psi (B, C_psi)

        with tf.variable_scope('res0_r'):
            out_r = adaIN(x, psi)
            out_r = relu(out_r)
            out_r = up_sampling_nn(out_r)
            out_r = conv(out_r, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        with tf.variable_scope('res1_r'):
            out_r = adaIN(out_r, psi)
            out_r = relu(out_r) 
            out_r = conv(out_r, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        
        with tf.variable_scope('res0_l'):
            out_l = up_sampling_nn(x)
            out_l = conv(out_l, channels, kernel=1, stride=1, pad=0, pad_type='zero', use_bias=use_bias, sn=sn)
        

        return out_r + out_l


def SelfAttention(x, sn=True, scope='attention'):
    # x (B, H, W, C)

    with tf.variable_scope(scope):
        b, h, w, c = x.get_shape().as_list()

        conv_f = conv(x, c // 8, kernel=1, stride=1, pad=0, sn=sn, scope='f_conv') # f (B, H, W, C/8)
        conv_g = conv(x, c // 8, kernel=1, stride=1, pad=0, sn=sn, scope='g_conv') # g (B, H, W, C/8)
        conv_h = conv(x, c, kernel=1, stride=1, pad=0, sn=sn, scope='h_conv') # h (B, H, W, C)

        f_flat = tf.reshape(conv_f, shape=[conv_f.shape[0], -1, conv_f.shape[-1]]) # f_flat (B, HW, C/8)
        g_flat = tf.reshape(conv_g, shape=[conv_g.shape[0], -1, conv_g.shape[-1]]) # g_flat (B, HW, C/8)
        h_flat = tf.reshape(conv_h, shape=[conv_h.shape[0], -1, conv_h.shape[-1]]) # h_flat (B, HW, C)
        
        s = tf.matmul(g_flat, f_flat, transpose_b=True) # s (B, HW, HW)
        attMap = tf.nn.softmax(s)

        out = tf.matmul(attMap, h_flat) # out (B, HW, C)
        out = tf.reshape(out, shape=x.get_shape()) # out (B, H, W, C)
        out = conv(out, c, kernel=1, stride=1, pad=0, sn=sn, scope='att_conv') # out (B, H, W, C)

        gamma = tf.get_variable('gamma', [1], initializer=tf.constant_initializer(0.0))
        
        return gamma * out + x 


def adaIN(x, psi, eps=1e-5, scope='adain'):
    with tf.variable_scope(scope):
        # x (B, H, W, C)
        # psi (B, psi_C)

        # instance normalization (x-u(x)) / sigma(x)
        mean = tf.reduce_mean(x, [1, 2], keepdims=True)
        var_inv = tf.rsqrt(tf.reduce_mean(x - mean, axis=[1, 2], keepdims=True) + eps)
        x = (x - mean) * var_inv
        

        # denormalization
        # x (B, 1, 1, C)
        channel = x.shape[-1] * 2 # channel (C * 2)
        
        style = fc(psi, channel) # style (B, C*2)
        style= tf.reshape(style, [-1, 2, 1, 1, x.shape[-1]]) # style (B ,2, 1, 1, C)

        gamma = style[:, 0] # gamma (B, 1, 1, C)
        beta = style[:, 1] # beta (B, 1, 1, C)

        return gamma * x + beta


if __name__ == '__main__':
    x = tf.random.normal([4, 6, 6, 8])
    psi = tf.random.normal([4, 512])

    x_shape = x.get_shape() 
    
    print('[Info] resblock adaIN')
    ResBlock_adaIN(x, psi, x_shape[-1])
    
    print('[Info] resblock down for generator')
    ResBlockDown_g(x, x_shape[-1])

    print('[Info] resblock down for discriminator')
    ResBlockDown_d(x, x_shape[-1])
    
    print('[Info] resblock up')
    ResBlockUp_adaIN(x, psi, x_shape[-1])
    
    print('[Info] self-attention')
    SelfAttention(x)

