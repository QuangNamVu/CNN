import numpy as np
import tensorflow as tf

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train.shape)

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training and test datasets
print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')

# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",  # index 1
                        "Pullover",  # index 2
                        "Dress",  # index 3
                        "Coat",  # index 4
                        "Sandal",  # index 5
                        "Shirt",  # index 6
                        "Sneaker",  # index 7
                        "Bag",  # index 8
                        "Ankle boot"]  # index 9

# Image index, you can pick any number between 0 and 59,999
img_index = 5
# y_train contains the lables, ranging from 0 to 9
label_index = y_train[img_index]
# # Print the label, for example 2 Pullover
# print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))
# # # Show one of the images from the training dataset
# plt.imshow(x_train[img_index])

N_train = 1000
N_val = 500

(x_train, x_valid) = x_train[:N_train], x_train[N_train:N_train + N_val]
(y_train, y_valid) = y_train[:N_train], y_train[N_train:N_train + N_val]

print(x_train.shape)

if len(x_train.shape) == 3:
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    x_valid = np.expand_dims(x_valid, axis=1)
    x_train = x_train / 255
    x_test = x_test / 255
    x_valid = x_valid / 255

from src_CNN.layers import *
from time import time

fastCNN = Model()

# Conv
fastCNN.add(Conv2DFast(filters=5, in_channel=1, kernel_size=5, stride=1, padding=2, learning_rate=0.01))

# ReLU
fastCNN.add(ReLU())

# MaxPool
fastCNN.add(MaxPoolingFast(pool_size=2, stride=1))

# FC
fastCNN.add(FullyConnected(hidden_dim=3645, num_classes=1024, learning_rate=0.01))

# DropOut
fastCNN.add(Dropout(0.5))

# FC
fastCNN.add(FullyConnected(hidden_dim=1024, num_classes=10))

t2 = time()
fastCNN.fit(x_train, y_train, x_valid, y_valid, epoch=10, batch_size=50, print_after=20)
t3 = time()

print("Time: ", (t3 - t2) / 60)
