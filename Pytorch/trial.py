import numpy as np
import matplotlib.pyplot as plt



# data preparation
train_images = np.load("./MNIST/npy-format-data/train_images.npy")/255
train_labels = np.load("./MNIST/npy-format-data/train_labels.npy")
test_images = np.load("./MNIST/npy-format-data/test_images.npy")/255
test_labels = np.load("./MNIST/npy-format-data/test_labels.npy")


idxs = np.where(train_labels==8)
train_images = train_images[idxs]
train_labels = train_labels[idxs]



col_size = 5
grid_imgs = np.vstack([ np.hstack([img for img in train_images[s:s+col_size]]) for s in range(0, col_size*5, col_size)])


plt.imshow(grid_imgs, cmap='gray')
plt.show()












