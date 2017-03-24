# all imports
import time
import pickle
import csv
import operator
from itertools import groupby

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from sklearn import model_selection

_index_in_epoch = 0
num_classes = 43
BATCH_SIZE = 2500
EPOCHS = 10000

def group_classes(Y):
    return {key:len(list(group)) for key, group in groupby(Y)}

def group_classes_sorted(Y):
    data = group_classes(Y)
    return sorted(data.items(), key=lambda x:x[1], reverse=True)


def plot_frequency(xlabel, xs, ys, with_names=True):
    fig, ax = plt.subplots(figsize=(15, 12))
    bars = ax.barh(xs, ys, 1, color='g', alpha=0.3)
    for i,bar in enumerate(bars):
        height = bar.get_y()
        if with_names:
            ax.text(bars[-1].get_width()-(bars[0].get_width()*6), height,
                '{} - {}'.format(i, xlabel[i]),rotation=0,ha='left', va='center')
        ax.text(bars[i].get_x()+bars[i].get_width()+10, height+bars[i].get_height()/2,
                '({} - {})'.format(i, ys[i]),rotation=0,ha='left', va='center')

    plt.show()
    
    
def get_images_and_counts(X, Y, count_data):
    images, labels, counts = [], [], []
    for label, count in count_data:
        images.append(X[Y.index(label)])
        counts.append(count)
        labels.append(label)

    return images, labels, counts


def plot_axes(axes, images, labels, counts=None, is_count=False, pred_labels=None):    
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap='binary')
        # Show true and predicted classes.
        if list(counts):
            xlabel = "Count: {0}".format(counts[i])
            title = "Class: {0}".format(labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], pred_labels[i])

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])


def plot_signs(images, labels, counts=None, pred_labels=None):
    """Create figure but watch out for 43!"""
    count = len(images)
    fig, axes = plt.subplots(6, 7, figsize=(10, 10))
    fig.subplots_adjust(hspace=1, wspace=1)
    plot_axes(axes, images[:-1], labels[:-1], counts, is_count=True)


def transform_image(img, ang_range, shear_range, trans_range):
    """
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    Copied from confluence post
    https://carnd-udacity.atlassian.net/wiki/display/CAR/Project+2+%28unbalanced+data%29+Generating+additional+data+by+jittering+the+original+image
    """
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def next_batch(data, labels, batch_size):
    """Return the next `batch_size` examples from this data set."""
    global _index_in_epoch
    start = _index_in_epoch
    _index_in_epoch += batch_size
    _num_examples = len(data)

    if _index_in_epoch > _num_examples:
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples

    end = _index_in_epoch
    return data[start:end], labels[start:end]

def inference(x_image, keep_prob):

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # convolution
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

    # X2 pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    x_image = tf.reshape(x_image, [-1, 32, 32, 1])

    # 1st conv 3x3x128
    # output: 32x32x128
    with tf.name_scope('conv1') as scope:
        w_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
        
    conv1 = tf.nn.relu(conv1)

    # 1st pooling 2x2
    # output : 18x18x512
    with tf.name_scope('pool1') as scope:
        pool1 = max_pool_2x2(conv1)

    # Flatten
    with tf.name_scope('fc1') as scope:
        fc1 = flatten(conv1)
        fc1_shape = (fc1.get_shape().as_list()[-1], 512)
        
        # (16 * 16 * 512, 120)
        fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape)))
        fc1_b = tf.Variable(tf.zeros(512))
        fc1 = tf.matmul(fc1, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

    #2nd fully connected
    with tf.name_scope('fc2') as scope:
        w_fc2 = weight_variable([512, 43])
        b_fc2 = bias_variable([43])

    # softmax output
    with tf.name_scope('softmax') as scope:
        y_conv = tf.matmul(fc1_drop, w_fc2) + b_fc2

    return y_conv

def loss(logits, labels):
    # cross entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    # TensorBoard
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    y_cls = tf.argmax(labels, dimension=1)
    y_pred = tf.argmax(logits, 1)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
    
    # TensorBoard
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

def normalize_greyscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    greyscale_min = 0
    greyscale_max = 255
    return a + ( ( (image_data - greyscale_min)*(b - a) )/( greyscale_max - greyscale_min ) )
    
"""
# Image Tensor
images_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')

gray = tf.image.rgb_to_grayscale(images_placeholder, name='gray')

gray /= 255.

# Label Tensor
labels_placeholder = tf.placeholder(tf.float32, shape=(None, 43), name='y')

# dropout Tensor
keep_prob = tf.placeholder(tf.float32, name='drop')

# construct model
logits = inference(gray, keep_prob)

# calculate loss
loss_value = loss(logits, labels_placeholder)

# training
train_op = training(loss_value, 0.001)

# accuracy
acc = accuracy(logits, labels_placeholder)
"""
