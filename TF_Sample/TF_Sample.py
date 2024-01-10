import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

x = tf.constant(4, shape = (1,1), dtype=tf.float32)
print(x)

x_ = tf.constant([1,2,3])
y_ = tf.constant([9,8,7])

z = tf.add(x_, y_)
z = tf.subtract(x_, y_)

