import tensorflow as tf
import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
from os import listdir
from os.path import isfile, join
import sys
import util

from model import BarkCNN

tf.logging.set_verbosity(tf.logging.INFO)

arguments = sys.argv

iterations = int(arguments[1])
paths_file = str(arguments[2])

train_data = util.getData(paths_file)
trainX, trainY = np.hsplit(train_data, [-1])

def trainNetConv(maxIter):

	myModel = BarkCNN()
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		saver = tf.train.Saver(tf.all_variables())
		myIters = 0
		tensors_to_log = {"probabilities": "sigmoid_tensor"}
		logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)
		while myIters < maxIter:
			sess.run(myModel.train_step,feed_dict={myModel.x: trainX, myModel.y_: trainY, myModel.keep_prob: 0.5})
			if myIters % 50 == 0:
				train_accuracy = myModel.accuracy.eval(session=sess,feed_dict={myModel.x:trainX, myModel.y_: trainY, myModel.keep_prob: 1.0})
				print("Step %d, Training accuracy: %g"%(myIters, train_accuracy))
			myIters+= 1
		save_path = saver.save(sess, "./model.ckpt")

trainNetConv(iterations)
