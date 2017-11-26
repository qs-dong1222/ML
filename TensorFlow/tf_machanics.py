import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist


# The MNIST dataset has 10 classes, representing the digits 0 through 9.
# NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
# IMAGE_SIZE = 28
# IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

data_sets = input_data.read_data_sets(" MNIST_data/", one_hot=True)
print(" mnist.IMAGE_PIXELS\n",mnist.IMAGE_PIXELS)
print("data_sets.validation\n",data_sets.validation)

