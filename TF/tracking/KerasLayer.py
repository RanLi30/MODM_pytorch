'''
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
import keras.utils
from tensorflow.keras.models import Model,load_model
'''
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

# create a tf session，and register with keras。
#sess = tf.Session()
#K.set_session(sess)
import tensorflow.keras as K
import tensorflow.keras.layers as KL
'''
class KerasLayer(Model):
    def __init__(self):
        super(KerasLayer,self).__init__(name='KerasLayer')
        #self.shapes=shapes
        self.conv1 = tensorflow.keras.layers.Conv2D(64,5,activation='relu')
        #self.flat = tensorflow.keras.layers.Flatten()
        #self.dense1 = tensorflow.keras.layers.Dense(256,activation='relu')
        #self.inputs = Input(shape=shapes,dtype='float32',name='inputs')

    def __call__(self, inputs, mask=None):
        #inputs = self.inputs
        x = self.conv1(inputs)
        #flat1 = self.flat(x)
        #x=self.dense1(flat1)

        return x
'''
img_width, img_height = 14, 14
input_shape = (img_width, img_height, 32)
#inputs = Input(shape=input_shape,dtype='float32',name='inputs')



class KerasLayer(K.Model):
    def __init__(self,input_model,name='KerasLayer'):
        x = input_model.outputs
        x = KL.Conv2D(1, 5,  padding="same",name='conv2')(x)
        super().__init__(inputs=input_model.outputs, outputs=x, name=name)
