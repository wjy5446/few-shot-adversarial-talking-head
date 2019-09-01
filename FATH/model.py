from collections import OrderedDict
import tensorflow as tf

from layers import *
from modules import *

class FATH(object):
    def __init__(self, sess=None, args=None):
        pass

    def __call__(self, ):
        pass

    def embedder(self, x, y, is_training=True, reuse=False):
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
            print(out)
            return out

    def generator(self, y, e, is_training=True, reuse=False):
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
                
                out = ResBlock_adaIN(out, front_style, back_style, 512, scope='resblock_' + str(i))
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

if __name__ == '__main__':
    x = tf.random_normal([4, 256, 256, 3])
    y = tf.random_normal([4, 256, 256, 3])
    
    fath = FATH()
    
    print("[INFO] embedder")
    e = fath.embedder(x, y)
   
    print("[INFO] generator")
    fath.generator(y, e)
