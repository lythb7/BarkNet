
import tensorflow as tf
import numpy as np 

class BarkCNN():
	def __init__(self):
		self.x = tf.placeholder(tf.float32, [None, 1024])
		self.y_ = tf.placeholder(tf.float32, [None, 1])

		self.x_image = tf.reshape(self.x, [-1,32,32,1])
		self.W_conv1 = weight_variable([5, 5, 1, 32])
		self.b_conv1 = bias_variable([32])
		self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1 = max_pool_2x2(self.h_conv1)
		self.W_conv2 = weight_variable([5, 5, 32, 64])
		self.b_conv2 = bias_variable([64])

		self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		self.h_pool2 = max_pool_2x2(self.h_conv2)
		self.W_fc1 = weight_variable([8 * 8 * 64, 1024])
		self.b_fc1 = bias_variable([1024])

		self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 8*8*64])
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
		self.keep_prob = tf.placeholder("float")
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
		self.W_fc2 = weight_variable([1024, 1])
		self.b_fc2 = bias_variable([1])
		self.h_fc2 = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
		self.y_conv=tf.nn.sigmoid(self.h_fc2)

		self.cross_entropy = tf.reduce_mean( -(self.y_*tf.log(self.y_conv)) - (1 - self.y_) * tf.log(1 - self.y_conv))
		self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
		self.correct_prediction = tf.equal(self.y_conv, self.y_)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')