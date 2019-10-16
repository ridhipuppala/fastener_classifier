from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2
from math import floor, ceil, pi
import re
from time import time
from time import sleep
from PIL import Image

IMAGE_SIZE = 28

def Testing(testX):
	ILN = 784
	HLN = 300
	beta = 0.01
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
	regularizers = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)

	train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)

	# initialize the graph
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	print("=========================== Loading Weights and Bias =================================")
	#d = 30
	d = 1
	k1 = d*1
	k2 = d*1
	k3 = d*1
	k4 = d*1
	k5 = d*2
	#k6 = d*1

	testY = [1]*k1 + [2]*k2 + [3]*k3 + [4]*k4 + [5]*k5 #+ [6]*k6

	yt = []
	for i in range(len(testY)):
		yt.append((np.arange(1,6)==testY[i]))
	at = np.reshape(np.asarray(yt),(-1,5))


	h = tf.nn.sigmoid(tf.matmul(x,W1)+b1)
	y = tf.nn.softmax(tf.matmul(h, W2) + b2)
	for i in range(150):
		sess.run(train_step, feed_dict={x: testX, y_: at})


	print("================= Predicting the Classes for Test Images ======================")
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

	prediction=tf.argmax(y,1)
	pred_lab = prediction.eval(feed_dict={x: testX}, session=sess)
	pred_lab[:] = pred_lab[:]  + 1
	#print("Predicted Labels: ", pred_lab)

	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#print("Testing Accuracy based on Test input labels: %.4f" % (sess.run(accuracy, feed_dict={x: testX, y_: at})))

	return pred_lab




def get_image_paths(folder):
    files = os.listdir(folder)
    files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
    #files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    return files


def cropping(folder):
	X_img_paths = get_image_paths(folder)
	i =0
	for path_img in X_img_paths:
		im = Image.open(path_img)
		#area = (1200, 1200, 3000, 2000) 
		area = (800, 1000, 3800, 2200) #original area  

		crop_img = im.crop(area)
		filename = './cropimgs_png/' + str(i) + '.png'
		crop_img.save(filename)
		#im.save(filename)
		i = i+1

def tf_resize_images(X_img_file_paths):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE), tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Each image is resized individually as different image may be of different size.
        for index, file_path in enumerate(X_img_file_paths):
            img = mpimg.imread(file_path)[:, :, :3] # Do not read alpha channel.
            resized_img = sess.run(tf_img, feed_dict = {X: img})
            X_data.append(resized_img)

    X_data = np.array(X_data, dtype = np.float32) 
    return X_data

def grayscale_conversion(final_imgs_color):
    resized_img_gray = []
    resized_img_color = np.zeros((IMAGE_SIZE,IMAGE_SIZE))
    for i in range(0,len(final_imgs_color)):
        final_img_color = final_imgs_color[i]
        for x in range(0,IMAGE_SIZE):
            for y in range(0,IMAGE_SIZE):
                resized_img_color[x][y] =  (final_img_color[x][y][0]+final_img_color[x][y][1] + final_img_color[x][y][2])/3.0
        resized_img = np.array(resized_img_color.flatten(),dtype = np.float32)
        resized_img_gray.append(resized_img)
    resized_img_gray = np.array(resized_img_gray,dtype = np.float32)
    return resized_img_gray




def main():

	folder = './testimages'   
	X_img_paths = get_image_paths(folder)
	print(X_img_paths)

	print(' ============================= Cropping the Test Images ================================')
	cropping(folder)

	print('=========Generating a single array of Test images after Resizing and Grayscale Conversion ==========')
	folder = './cropimgs_png'   
	X_img_paths = get_image_paths(folder)
	print(X_img_paths)
	X_imgs = tf_resize_images(X_img_paths)
	final_array = grayscale_conversion(X_imgs)
	print('Dimensions of final array for Feedforward Input (Prediction)')
	print(final_array.shape)

	labels = Testing(final_array)
	i =0
	label_dict = {1:'Bearing',2:'Counter Sunkbolt',3:'Hexbolt',4:'Hexnut',5:'Unknown',6:'Unknown'}
	for filename in X_img_paths:

		img = cv2.imread(str(filename))
		a,b,ch = img.shape
		k = labels[i]
		countour_img = img.copy()
		gray = cv2.cvtColor(countour_img, cv2.COLOR_BGR2GRAY)
		gray = cv2.medianBlur(gray,5)
		gray = cv2.dilate(gray, None, iterations=10)
		gray = cv2.erode(gray, None, iterations=10)
		th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,3)
		th3 = cv2.adaptiveThreshold(th2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)
		ims, cnts, hierarchys = cv2.findContours(th3.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(countour_img, cnts, -1, (0,255,0), 5)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(countour_img,'%s(id=%d)'%(label_dict[k],k), (int(0.4*b),int(0.1*a)), 4, 3, ( 255,0,0), 2, cv2.LINE_AA)
		cv2.namedWindow('Test Image %d' %i, cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Test Image %d' %i , 500,500)
		cv2.imshow('Test Image %d' %i,countour_img)
		savepath = './save/' + str(i) + '.png'
		cv2.imwrite(savepath,countour_img)
		i = i+1

		# font = cv2.FONT_HERSHEY_SIMPLEX
		# cv2.putText(img,'%s(id=%d)'%(label_dict[k],k), (int(0.4*b),int(0.1*a)), 4, 3, ( 255,0,0), 2, cv2.LINE_AA)
		# cv2.namedWindow('Test Image %d' %i, cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('Test Image %d' %i , 500,500)
		# cv2.imshow('Test Image %d' %i,img)
		# savepath = './save/' + str(i) + '.png'
		# cv2.imwrite(savepath,img)
		# i = i+1


	k = cv2.waitKey(0) & 0xFF
	if k == 27:
	# wait for ESC key to exit
		pass
		cv2.destroyAllWindows()













if __name__== "__main__":
  main()