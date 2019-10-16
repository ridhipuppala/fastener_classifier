#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from time import time
import numpy as np
  
# Download the dataset
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
itr = 15
HLN = 300
alpha = 0.001
Epochs = 100
performancedata = np.zeros((itr,5))
for param in range(0,itr):
	X = np.load('dataset.npy')
	Y = np.load('class_id.npy')
	y = []
	for i in range(len(Y)):
		y.append((np.arange(1,6)==Y[i]))
	a = np.reshape(np.asarray(y),(-1,5))
	#print(a.shape)

	#Y = ()
	x = tf.placeholder(tf.float32, [None, 784])
	W1 = tf.Variable(tf.random_normal([784,HLN]))
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

	# define a training step
	train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)

	# initialize the graph
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	# train
	#print("Starting the training...")
	start_time = time()
	for i in range(Epochs):
	    #batch_xs, batch_ys = mnist.train.next_batch(100)

	    sess.run(train_step, feed_dict={x: X, y_: a})
	#print("The training took %.4f seconds." % traintime)

	# validate
	traintime = float((time() - start_time))
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	acc_value = (sess.run(accuracy, feed_dict={x: X, y_: a}))
	#print("Accuracy: %.4f" % (sess.run(accuracy, feed_dict={x: X, y_: a})))
	print("Hidden Layer Neurons: %d 	Learn Rate: %.4f 	Epochs: %d  Train Time: %.4f 	Accuracy: %.4f" %(HLN,alpha,Epochs,traintime,acc_value))
	performancedata[param,:] = [HLN,alpha,Epochs,traintime,acc_value]
	alpha = alpha + 0.001
print(performancedata)
np.save('performance.npy',performancedata,allow_pickle = True,fix_imports = True)
np.savetxt("performance.csv", performancedata, delimiter=",")