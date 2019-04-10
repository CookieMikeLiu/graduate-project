import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imread
import matplotlib
from tensorflow.python.framework import ops
import os
import glob

pos_im_path = './images/pos_person'
neg_im_path = './images/neg_person'
test_path = './test_image'

def load_dataset():

    train_set_x_orig = []
    train_set_y_orig = []
    test_set_x_orig = []
    test_set_y_orig = [0,0,0,0,0,0,1,1,1,1,1,1]
    classes = [0,1]

    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        im = matplotlib.image.imread(im_path)
        train_set_x_orig.append(im)
        train_set_y_orig.append(1)
    for im_path in glob.glob(os.path.join(neg_im_path,'*')):
        im = matplotlib.image.imread(im_path)
        train_set_x_orig.append(im)
        train_set_y_orig.append(0)

    for im_path in glob.glob(os.path.join(test_path,'*')):
        im = matplotlib.image.imread(im_path)
        test_set_x_orig.append(im)

    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_orig = np.array(test_set_x_orig)
    test_set_y_orig = np.array(test_set_y_orig)
    classes = np.array(classes)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X),b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1),b2)                     # Z2 = np.dot(W2, a1) + b2


    return Z2

def predict(X, parameters):

    W1 = tf.convert_to_tensor(parameters["W1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    params = {"W1": W1,
              "W2": W2,
              "b1":b1,
              "b2":b2}

    x = tf.placeholder("float", [32400, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})

    return prediction

if __name__ == '__main__':
    X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes= load_dataset()
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    Y_train = convert_to_one_hot(Y_train_orig, 2).T
    Y_test = convert_to_one_hot(Y_test_orig, 2).T
    a = random_mini_batches(X_train,Y_train)
