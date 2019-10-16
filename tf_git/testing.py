from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from time import time
import numpy as np
  
# Download the dataset
#from tensorflow.examples.tutorials.mnist import input_data
testX = np.load('testdataset_augmented.npy')
testY = np.load('testclass_id.npy')
y = []
for i in range(len(Y)):
	y.append((np.arange(1,6)==Y[i]))
a = np.reshape(np.asarray(y),(-1,5))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define a training step
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# initialize the graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

sess.run(train_step, feed_dict={x: testX, y_: a})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: %.4f" % (sess.run(accuracy, feed_dict={x: X, y_: a}))

