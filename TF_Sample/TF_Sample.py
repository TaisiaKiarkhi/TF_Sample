#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
#import tensorflow as tf
#from tensorflow import keras 
#from keras import layers
#from keras.datasets import mnist
#
#
#x = tf.constant(4, shape = (1,1), dtype=tf.float32)
#print(x)
#
#x_ = tf.constant([1,2,3])
#y_ = tf.constant([9,8,7])
#
#z = tf.add(x_, y_)
#z = tf.subtract(x_, y_)
#
#print(z)
#
##physical_devices = tf.config.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
#
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
#x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
#
##Sequential API (conveniet, not very flexible)
#
#model = keras.Sequential(
#    [
#        keras.Input(shape=(28*28)),
#        layers.Dense(512, activation = 'relu'),
#        layers.Dense(256, activation = 'relu'),
#        layers.Dense(10),
#        ])
#
#print(model.summary())
#
#import sys
#sys.exit()
#model.compile(
#    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
#    optimizer = keras.optimizer.Adam(lr = 0.001),
#    metrics = ["accuracy"],
#    
#    )
#
#model.fit(x_train, y_train, batch_size = 32, epochs = 5, verbose = 2)
#model.evaluate(x_test, y_test, batch_size=32, verbose =2)
#
#


import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf 
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

physical_devices = tf.config.list_physical_devices()
tf.config.experimental.set_memory_growth(physical_devices[1], True)

(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_train.reshape(-1, 28*28).astype("float32") / 255.0

#very convinient, not flexible

model = keras.Sequential(
    [
        keras.Input(shape=28*28),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizer.Adam(lr = 0.001),
    metrics = ["accuracy"],
    
    ) 

