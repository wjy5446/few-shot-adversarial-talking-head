import tensorflow as tf

from module import *

class FATH(object):
    def __init__(self, sess, args):
        pass

    def __call__(self, ):
        pass

    def embedder(self, x, is_training=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):

        pass

    def generator(self, x, is_training=True, reuse=False):
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

