from collections import OrderedDict
import tensorflow as tf
import keras

from keras.applications.vgg19 import preprocess_input
from keras_vggface.vggface import VGGFace

from layers import *
from loss import *
from modules import *


class FATH(object):
    def __init__(self, sess=None, args=None):
        self.model_name = 'FATH'
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.k = args.k
        self.img_size = args.img_size
        self.c_dim = args.c_dim
        self.e_dim = args.e_dim

        self.smaple_num = args.sample_num
        self.test_num = args.test_num

        # custom
        self.n_video = args.n_video

        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        # dataset
        self.data = ?
        self.dataset_num = len(self.data)

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        
        # model
        self.embedder = Embedder('embedder')
        self.generator = Generator('generator')
        self.discriminator = Discriminator('discriminator')
        self.vgg = Vgg19()
        self.vggface = VggFace()

    def __call__(self, ):
        pass

    def build_model(self):
        """ Graph model"""
       
        self.inputs_image = tf.placeholder(tf.float32, [self.batch_size, self.k + 1, self.img_size, self.img_size, self.c_dim], name='real_images')
        self.inputs_landmark = tf.placeholder(tf.float32, [self.batch_size, self.k + 1, self.img_size, self.img_size, self.c_dim], name='real_landmarks')
        self.index = tf.placeholder(tf.float32, [self.batch_size], name='real_index')
        

        """graph flow"""
        images = tf.slice(self.inputs_image, [0, 0, 0, 0, 0], [-1, K, -1, -1, -1])
        landmarks = tf.slice(self.inputs_landmark, [0, 0, 0, 0, 0], [-1, K, -1, -1, -1])
        t_images = tf.slice(self.inputs_image, [0, K, 0, 0, 0], [-1, 1, -1, -1, -1])
        t_landmarks = tf.slice(self.inputs_landmark, [0, K, 0, 0, 0], [-1, 1, -1, -1, -1])
        
        x = tf.reshape(images, [self.batch_size * self.k, self.image_size, self.image_size, self.c_dim])
        y = tf.reshape(landmarks, [self.batch_size * self.k, self.image_size, self.image_size, self.c_dim])
        
        x_t = tf.reshape(t_images, [self.batch_size, self.image_size, self.image_size, self.c_dim])
        y_t = tf.reshape(t_landmarks, [self.batch_size, self.image_size, self.image_size, self.c_dim])

        e_vectors = self.embedder(x, y)
        e_vectors = tf.reshape(e_vector, [self.batch_size, self.k, self.e_dim])
        e = tf.reduce_mean(e_vectors, axis=1)

        # generate fake image
        x_fake = self.generator(y_t, e, self.index)
        
        score_fake, li_fm_fake = self.discriminator(x_fake, y_t, self.index)
        score_real, li_fm_real = self.discriminator(x_t, y_t, self.index, reuse=True)

        with tf.variable_scope('discriminator', reuse=True):
            with tf.variable_scope('projection'):
                W = tf.get_variable('W')

        loss_cnt = LossCnt(x_t, x_fake, self.vgg19, self.vggface) 
        loss_adv = LossAdv(score_fake, li_fm_real, li_fm_fake) 
        loss_match = LossMatch(e, W, self.index)
        loss_d = LossD(score_real, fake_score)
        loss_G = loss_cnt + loss_adv + loss_match
        loss_D = loss_d

    
        # get trainable variable
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'discriminator' in var.name]
        d_vars = [var for var in t_vars if 'generator' in var.name]

        self.g_optim = tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(loss_G, var_list=g_vars)
        self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(loss_D, var_list=d_vars)
        


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
             
            # ResBlock
            for i in range(0, 5):
                front_style = (tf.slice(e_psi, [0, style_info[(4*i)+0][0]], [-1, style_info[(4*i)+0][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+1][0]], [-1, style_info[(4*i)+1][1]]))
                back_style = (tf.slice(e_psi, [0, style_info[(4*i)+2][0]], [-1, style_info[(4*i)+2][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+3][0]], [-1, style_info[(4*i)+3][1]]))
                
                out = ResBlock_adaIN(out, front_style, back_style, 512, scope='resblock_' + str(i))
            
            # Upsampling
            ch = 512
            for i in range(5, 9): 
                front_style = (tf.slice(e_psi, [0, style_info[(4*i)+0][0]], [-1, style_info[(4*i)+0][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+1][0]], [-1, style_info[(4*i)+1][1]]))
                back_style = (tf.slice(e_psi, [0, style_info[(4*i)+2][0]], [-1, style_info[(4*i)+2][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+3][0]], [-1, style_info[(4*i)+3][1]]))
                 
                out = ResBlockUp_adaIN(out, front_style, back_style, ch, scope='front_resblock_up_' + str(i - 5))
                
                if i > 5:
                    ch = ch // 2

            out = SelfAttention(out, scope='attention_1')
            print(out)
            for i in range(9, 11):
                front_style = (tf.slice(e_psi, [0, style_info[(4*i)+0][0]], [-1, style_info[(4*i)+0][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+1][0]], [-1, style_info[(4*i)+1][1]]))
                back_style = (tf.slice(e_psi, [0, style_info[(4*i)+2][0]], [-1, style_info[(4*i)+2][1]]), 
                              tf.slice(e_psi, [0, style_info[(4*i)+3][0]], [-1, style_info[(4*i)+3][1]]))
                out = ResBlockUp_adaIN(out, front_style, back_style, ch, scope='back_resblock_up_' + str(i - 5))
                ch = 3
           
            # activate
            out = relu(out)
            
            return out 

################################################
# Discriminator
################################################

class Discriminator:
    def __init__(self, name):
        self.name = name
        self.n_video = 10

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

                out = SelfAttention(out)
                
                for i in range(0, 3):
                    out = ResBlockDown_d(out, 512, scope='resblock_down_back_' + str(i))
                    interm_feature.append(out)

                out = ResBlock(out, 512, scope='resblock')

                out = global_sum_pooling(out) # out (B, 512)
                out = relu(out) 
                out = tf.expand_dims(out, 1) 

            with tf.variable_scope('projection', reuse=reuse):
                W = tf.get_variable('W', [self.n_video, 512], dtype=tf.float32, initializer=tf.random_normal_initializer())
                w0 = tf.get_variable('w0', shape=[1, 512], dtype=tf.float32, initializer=tf.random_normal_initializer())
                b = tf.get_variable('b', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                
                print(W)

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
        
        self.slice1 = keras.Sequential()
        self.slice2 = keras.Sequential()
        self.slice3 = keras.Sequential()
        self.slice4 = keras.Sequential()
        self.slice5 = keras.Sequential()

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

    
    print("[INFO] embedder")
    embedder = Embedder('embedder')
    e = embedder(x, y)

    print("[INFO] generator")
    generator = Generator('generator')
    fake = generator(y, e)    

    print("[INFO] discriminator")
    discriminator = Discriminator('discriminator')
    r_real, fm_real = discriminator(x, y, i)
    
    r_fake, fm_fake = discriminator(fake, y, i, reuse=True)

    print("[INFO] VGG")
    vgg19 = Vgg19()
    out = vgg19(x)

    print("[INFO] VGGFACE")
    vggface = VggFace()
    out = vggface(x)

    print('[INFO] LossCnt')
    loss1 = LossCnt(x, fake, vgg19, vggface)
    print(loss1)

    print('[INFO] LossAdv')
    loss2 = LossAdv(r_fake, fm_real, fm_fake)
    print(loss2)

    print('[INFO] Loss Match')
    with tf.variable_scope('discriminator', reuse=True):
        with tf.variable_scope('projection'):
            W = tf.get_variable('W')

    print('[INFO] Loss D')
    loss4 = LossD(r_real, r_fake)
    print(loss4)
