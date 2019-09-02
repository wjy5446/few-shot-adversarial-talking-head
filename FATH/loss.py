import tensorflow as tf
from keras.applications.vgg19 import preprocess_input
from keras_vggface.vggface import VGGFace

from layers import *

#########################
# Generator Loss
#########################

def LossCnt(real, fake, vgg, vggface, weight_vgg=0.01, weight_vggface=0.001):
    list_real_vgg, list_fake_vgg = vgg(real), vgg(fake)
    list_real_vggface, list_fake_vggface = vggface(real), vggface(fake)

    loss_vgg = 0
    for real_vgg, fake_vgg in zip(list_real_vgg, list_fake_vgg):
        loss_vgg += tf.reduce_mean(tf.abs(real_vgg - fake_vgg))

    loss_vggface = 0
    for real_vggface, fake_vggface in zip(list_real_vggface, list_fake_vggface):
        loss_vggface += tf.reduce_mean(tf.abs(real_vggface - fake_vggface))
    

    return (weight_vgg * loss_vgg) + (weight_vggface * loss_vggface)

def LossAdv(fake_score, list_real_FM, list_fake_FM, weight_FM=10):    
    
    lossFM = 0
    for real_FM, fake_FM in zip(list_real_FM, list_fake_FM):
        lossFM += tf.reduce_mean(tf.abs(real_FM - fake_FM))

    return -tf.reduce_mean(fake_score) + (weight_FM * lossFM)     

def LossMatch(e, W, idx, weight_mch=80):
    W_i = tf.gather(W, idx)
    return tf.reduce_mean(tf.abs(e - W_i)) * weight_mch

#########################
# Discriminator Loss
#########################

def LossD(real_score, fake_score):
    real_score = tf.reduce_mean(real_score)
    fake_score = tf.reduce_mean(fake_score)
    
    return (relu(1 + fake_score) + relu(1 - real_score))

if __name__ == "__main__":
    pass

