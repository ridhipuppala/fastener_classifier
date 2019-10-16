from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from time import time
import numpy as np
from time import sleep
  
X = np.load('dataset.npy')
Y = np.load('class_id.npy')
y = []
for i in range(len(Y)):
	y.append((np.arange(1,6)==Y[i]))
a = np.reshape(np.asarray(y),(-1,5))
print(a.shape)

#Y = ()
ILN = 784
HLN = 300
#beta = 0.01
alpha = 0.005

x = tf.placeholder(tf.float32, [None, ILN])
W1 = tf.Variable(tf.random_normal([ILN,HLN]))
b1 = tf.Variable(tf.random_normal([HLN]))
W2 = tf.Variable(tf.random_normal([HLN, 5]))
b2 = tf.Variable(tf.random_normal([5]))



# define the model
h = tf.nn.sigmoid(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

# correct labels
y_ = tf.placeholder(tf.float32, [None, 5])


# define the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#regularizers = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)
# define a training step
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)

# initialize the graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#save_path = saver.save(sess, "./model.ckpt")

# train
'''
print("Starting the training...")
start_time = time()
for i in range(3):
    #batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict={x: X, y_: a})
print("The training took %.4f seconds." % (time() - start_time))

# validate
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Training Accuracy: %.4f" % (sess.run(accuracy, feed_dict={x: X, y_: a})))
print('training Done')
'''

print("Starting testing...")
testX = np.load('testdataset_raw.npy')
#testX = np.load('shit1.npy')
#d = 30
d = 1
k1 = d*3
k2 = d*3
k3 = d*2
k4 = d*4
k5 = d*2
#testY = np.load('testclass_id.npy')
testY = [1]*k1 + [2]*k2 + [3]*k3 + [4]*k4 + [5]*k5
#testY = [1]*1 + [3]*1 + [1]*2
#print(testY)

yt = []
for i in range(len(testY)):
	yt.append((np.arange(1,6)==testY[i]))
at = np.reshape(np.asarray(yt),(-1,5))


h = tf.nn.sigmoid(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)
for i in range(12):
	sess.run(train_step, feed_dict={x: testX, y_: at})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

prediction=tf.argmax(y,1)
print("Predictions: ", prediction.eval(feed_dict={x: testX}, session=sess))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Complete Testing Accuracy: %.4f" % (sess.run(accuracy, feed_dict={x: testX, y_: at})))

