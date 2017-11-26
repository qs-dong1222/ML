import tensorflow as tf
import numpy as np
import cv2
import opencvlib
from tensorflow.examples.tutorials.mnist import input_data


featureSet = np.load("features.npy");
labelSet = np.load("labels.npy");

W = tf.Variable(tf.zeros([featureSet.shape[1], labelSet.shape[1]]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1, labelSet.shape[1]]), dtype=tf.float32)

sess = tf.Session();
tf.train.Saver().restore(sess, "./PlateRecogNet/PlateRecogNet.ckpt")

print("W_save",sess.run(W),"b_save",sess.run(b))


# predict_outputs = tf.nn.softmax(tf.matmul(Xholder, W) + b)
