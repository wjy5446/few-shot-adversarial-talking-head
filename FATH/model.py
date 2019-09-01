from collections import OrderedDict
import tensorflow as tf

from keras.applications.vgg19 import preprocess_input
from keras_vggface.vggface import VGGFace

from layers import *
from loss import *
from modules import *


class FATH(object):
    def __init__(self, sess=None, args=None):
        self.n_video = 10


    def __call__(self, ):
        pass

    def build_model(self):
        
        self.embedder = Embedder('embedder')
        self.generator = Generator('generator')
        self.discriminator = Discriminator('discriminator')
        self.vgg = Vgg19()
        self.vggface = VggFace()
        pass

    def train(self):
        pass
    
    def test(self):
        pass

    def load(self, checkpoint_dir):
        pass

    def visualize_result(self, epoch):
        pass

################################################
# Embedder
################################################

class Embedder:
    def __init__(self, name):
        self.name = name

    def __call__(self, x, y, is_training=True, reuse=False):
        # x (B, 256, 256, 3)
        # y (B, 256, 256, 3)
        with tf.variable_scope('embedder', reuse=reuse): 
            ch = 64
            max_ch = 512
            
            out = tf.concat([x, y], -1) # out (B, 256, 256, 6)
            
            for i in range(0, 3):
                out = ResBlockDown_d(out, ch, scope='front_resblock_' + str(i))
                ch = ch * 2

            out = SelfAttention(out, scope='attention_0') 

            for i in range(3, 6):
                out = ResBlockDown_d(out, ch, scope='back_resblock_' + str(i))

            out = tf.squeeze(global_sum_pooling(out)) # out (B, 512)
            out = relu(out) # out (B, 512)
            return out

################################################
# Generator
################################################

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, y, e, is_training=True, reuse=False):
        # y (B, 256, 256, 3)
        # e (B, 512)
        
        # projection (B, 512, style_len)

        DICT_STYLE_DIM = OrderedDict([
            ('res0' , {'front': 512, 'back': 512}),
            ('res1' , {'front': 512, 'back': 512}),
            ('res2' , {'front': 512, 'back': 512}),
            ('res3' , {'front': 512, 'back': 512}),
            ('res4' , {'front': 512, 'back': 512}),
            ('resup0' , {'front': 512, 'back': 512}),
            ('resup1' , {'front': 512, 'back': 512}),
            ('resup2' , {'front': 512, 'back': 256}),
            ('resup3' , {'front': 256, 'back': 128}),
            ('resup4' , {'front': 128, 'back': 64}),
            ('resup5' , {'front': 64, 'back': 3}),
        ])
            
        style_info = []
        for layer in DICT_STYLE_DIM:
            info = DICT_STYLE_DIM[layer]
            
            if len(style_info) == 0:
                style_info.append((0, info['front']))
            else:
                style_info.append((style_info[-1][0] + info['front'], info['front']))

            style_info.append((style_info[-1][0] + info['front'], info['front']))
            style_info.append((style_info[-1][0] + info['back'], info['back']))
            style_info.append((style_info[-1][0] + info['back'], info['back']))

        with tf.variable_scope('generator', reuse=reuse):
            B = y.get_shape().as_list()[0]
            
            # psi
            p = tf.get_variable('projection', [1, style_info[-1][0], 512], dtype=tf.float32, 
                                initializer=tf.random_normal_initializer(0.0, 0.02)) # p (1, len_style, 512)
         
            p = tf.tile(p, [B, 1, 1]) # p (B, len_style, 512)
            e = tf.expand_dims(e, 2)
            e_psi = tf.matmul(p, e) 
            e_psi = tf.squeeze(e_psi) 
            
            # Downsampling 
            ch = 64
           
            out = y
            for i in range(0, 3): 
                out = ResBlockDown_g(out, ch, scope='front_resblock_down_' + str(i))
                ch = ch * 2

            out = SelfAttention(out, scope='attention_0')
            
            for i in range(3, 6):
                out = ResBlockDown_g(out, ch, scope='back_resblock_down_' + str(i))
            
            print('--- [INFO] Downsampling')
            print(out)
            print(tf.slice(e_psi, [0, style_info[0][0]], [-1, style_info[0][1]]))
           
            print('--- [INFO] Resblock')
            # ResBlock
            for i in range(0, 5):
                front_style = (tf.slice(e_psi, [0, style_info[(4*i)+0][0]], [-1, style_info[(4*i)+0][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+1][0]], [-1, style_info[(4*i)+1][1]]))
                back_style = (tf.slice(e_psi, [0, style_info[(4*i)+2][0]], [-1, style_info[(4*i)+2][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+3][0]], [-1, style_info[(4*i)+3][1]]))
                
                out = ResBlock_adaIa(out, front_style, back_style, 512, scope='resblock_' + str(i))
                print(front_style, back_style)
                print(out)
           
            print('--- [INFO] Upsampling')
            # Upsampling
            ch = 512
            for i in range(5, 9): 
                front_style = (tf.slice(e_psi, [0, style_info[(4*i)+0][0]], [-1, style_info[(4*i)+0][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+1][0]], [-1, style_info[(4*i)+1][1]]))
                back_style = (tf.slice(e_psi, [0, style_info[(4*i)+2][0]], [-1, style_info[(4*i)+2][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+3][0]], [-1, style_info[(4*i)+3][1]]))
                 
                out = ResBlockUp_adaIN(out, front_style, back_style, ch, scope='front_resblock_up_' + str(i - 5))
                print(front_style, back_style)
                print(out)
                if i > 5:
                    ch = ch // 2

            out = SelfAttention(out, scope='attention_1')
            print(out)
            for i in range(9, 11):
                front_style = (tf.slice(e_psi, [0, style_info[(4*i)+0][0]], [-1, style_info[(4*i)+0][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+1][0]], [-1, style_info[(4*i)+1][1]]))
                back_style = (tf.slice(e_psi, [0, style_info[(4*i)+2][0]], [-1, style_info[(4*i)+2][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+3][0]], [-1, style_info[(4*i)+3][1]]))
                ch = 3
                print(front_style, back_style)
                print(out)
           
            # activate
            out = relu(out)
            
            return out 

################################################
# Discriminator
################################################

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, x, y, idx, is_training=True, reuse=False): 
        # x (B, 256, 256, 3)
        # y (B, 256, 256, 3)
        # idx (B)

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv_part', reuse=reuse):
                out = tf.concat([x, y], -1)
                print(out)
                
                ch = 64
                interm_feature = []
                for i in range(0, 3):
                    out = ResBlockDown_d(out, ch, scope='resblock_down_front_' + str(i))
                    interm_feature.append(out)
                    ch = ch * 2
                    print(out)

                out = SelfAttention(out)
                print(out)
                
                for i in range(0, 3):
                    out = ResBlockDown_d(out, 512, scope='resblock_down_back_' + str(i))
                    interm_feature.append(out)
                    print(out)

                out = ResBlock(out, 512, scope='resblock')

                out = global_sum_pooling(out) # out (B, 512)
                out = relu(out) 
                out = tf.expand_dims(out, 1) 
                print(out)

            with tf.variable_scope('projection', reuse=reuse):
                W = tf.get_variable('W', [self.n_video, 512], dtype=tf.float32, initializer=tf.random_normal_initializer())
                w0 = tf.get_variable('w0', shape=[1, 512], dtype=tf.float32, initializer=tf.random_normal_initializer())
                b = tf.get_variable('b', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                
                W_i = tf.gather(W, idx)
                W = tf.expand_dims(W_i + w0, 2)
               
                out = tf.matmul(out, W) + b # out (B)
                out = sigmoid(out)
                out = tf.squeeze(out)
                return out, interm_feature 


################################################
# VGG19
################################################

class Vgg19(tf.keras.Model):
    def __init__(self, trainable=False):
        super(Vgg19, self).__init__(name='Vgg19')
        vgg_pretrained_features = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False)

        if trainable is False:
            vgg_pretrained_features.trainable = False

        vgg_pretrained_features = vgg_pretrained_features.layers
        print(len(vgg_pretrained_features))
        
        self.slice1 = tf.keras.Sequential()
        self.slice2 = tf.keras.Sequential()
        self.slice3 = tf.keras.Sequential()
        self.slice4 = tf.keras.Sequential()
        self.slice5 = tf.keras.Sequential()
       
        for x in range(1, 2):
            self.slice1.add(vgg_pretrained_features[x]) 
        for x in range(2, 5):
            self.slice2.add(vgg_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add(vgg_pretrained_features[x])
        for x in range(8, 13):
            self.slice4.add(vgg_pretrained_features[x])
        for x in range(13, 18):
            self.slice5.add(vgg_pretrained_features[x])
                
    def call(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
         
        return out


################################################
# VGGFace
################################################

class VggFace(tf.keras.Model):
    def __init__(self, trainable=False):
        super(VggFace, self).__init__(name='VggFace')
        vgg_pretrained_features = VGGFace(include_top=False, input_shape=[224, 224, 3], pooling='avg')
        
        if trainable is False:
            vgg_pretrained_features.trainable = False
        
        vgg_pretrained_features = vgg_pretrained_features.layers
        print(vgg_pretrained_features)
        
        self.slice1 = tf.keras.Sequential()
        self.slice2 = tf.keras.Sequential()
        self.slice3 = tf.keras.Sequential()
        self.slice4 = tf.keras.Sequential()
        self.slice5 = tf.keras.Sequential()

        for x in range(1, 2):
            self.slice1.add(vgg_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add(vgg_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add(vgg_pretrained_features[x])
        for x in range(8, 12):
            self.slice4.add(vgg_pretrained_features[x])
        for x in range(12, 16):
            self.slice5.add(vgg_pretrained_features[x])

    def call(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
         
        return out

if __name__ == '__main__':
    x = tf.random_normal([4, 256, 256, 3])
    y = tf.random_normal([4, 256, 256, 3])
    i = [0, 1, 2, 3]   

    fath = FATH()
    
    print("[INFO] embedder")
    #e = fath.embedder(x, y)
   
    print("[INFO] generator")
    #fath.generator(y, e)

    print("[INFO] discriminator")
    fath.discriminator(x, y, i)
