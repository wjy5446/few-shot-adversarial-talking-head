import tensorflow as tf

from layers import *
from modules import *

class FATH(object):
    def __init__(self, sess, args):
        pass

    def __call__(self, ):
        pass

    def embedder(self, x, y, is_training=True, reuse=False):
        # x (B, 256, 256, 3)
        # y (B, 256, 256, 3)
        with tf.variable_scope('embedder', reuse=reuse): 
            out = tf.concat([x, y], -1) # out (B, 256, 256, 6)
            out = ResBlockDown_d(out, 64) # out (B, 128, 128, 64)
            out = ResBlockDown_d(out, 128) # out (B, 64, 64, 128)
            out = ResBlockDown_d(out, 256) # out (B, 32, 32, 256)
            out = SelfAttention(out) # out (B, 32, 32, 256)
            out = ResBlockDown_d(out, 512) # out (B, 16, 16, 512)
            out = ResBlockDown_d(out, 512) # out (B, 8, 8, 512)
            out = ResBlockDown_d(out, 512) # out (B, 4, 4, 512)
            out = tf.squeeze(global_sum_pooling(out)) # out (B, 512)
            out = relu(out) # out (B, 512)

            return out

    def generator(self, y, e, is_training=True, reuse=False):
        # y (B, 256, 256, 3)
        # e (B, 512)

        DICT_ADAIN_LAYER = Orderdict([
            ('res0', (512, 512))
            ('res1', (512, 512))
            ('res2', (512, 512))
            ('res3', (512, 512))
            ('res4', (512, 512))
            ('res_down0', (512, 512))
            ('res_down1', (512, 512))
            ('res_down2', (512, 256))
            ('res_down3', (256, 128))
            ('res_down4', (128, 64))
            ('res_down5', (64, 3))
        ])



        with tf.variable_scope('generator', reuse=reuse):
            out = ResBlockDown_g(y, 64) # out (B, 128, 128, 64)
            out = ResBlockDown_g(out, 128) # out (B, 64, 64, 128)
            out = ResBlockDown_g(out, 256) # out (B, 32, 32, 256)
            out = SelfAttention(out) # out (B, 32, 32, 256)
            out = ResBlockDown_g(out, 512) # out (B, 16, 16, 512)
            out = ResBlockDown_g(out, 512) # out (B, 8, 512)
            out = ResBlockDown_g(out, 512) # out (B, 4, 4, 512)

            
        pass

    def discriminator(self, x, reuse=False):
        pass

    def build_model(self):
        pass

    def train(self):
        pass
    
    def test(self):
        pass

    def save(self, checkpoint_dir, step):
        pass

    def load(self, checkpoint_dir):
        pass

    def visualize_result(self, epoch):
        pass

