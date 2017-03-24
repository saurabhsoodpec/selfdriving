# Load pickled data
import pickle

# TODO: fill this in based on where you saved the training and testing data
training_file = "../traffic-signs-data/train.p"
testing_file = "../traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

### To start off let's do a basic data summary.

# TODO: number of training examples
n_train = len(train['features'])

# TODO: number of testing examples
n_test = len(test['features'])

# TODO: what's the shape of an image?
sample_image_shape = train['sizes'][0]

# TODO: how many classes are in the dataset
n_classes = len(set(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", sample_image_shape)
print("Number of classes =", n_classes)

print("Length features,labels,sizes  =", len(train['features']),"," ,len(train['labels']),",", len(train['sizes']))

#print("1st feature =", train['features'][0])
#print("1st size =", train['sizes'][0])
#print("1st label =", train['labels'][0])
#print("1st coords =", train['coords'][0])

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import scipy as scipy
import math
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

tf.__version__

# Problem 1 - Implement Min-Max scaling for greyscale image data
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
    
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    	

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
#enc = preprocessing.OneHotEncoder()
lb.fit(train['labels'])
#print(lb.transform(train['labels']))
train['hot-labels'] = lb.transform(train['labels'])
test['hot-labels'] = lb.transform(test['labels'])

train["input-features"] = []
test["input-features"] = []
train["input-features-noresize"] = []
test["input-features-noresize"] = []

image_size_tuple = (32,32,3)
image_size_flat = 32*32*3

for i, option in enumerate(train['features']):
    #print('Size - ', train['sizes'][i])
    #print('Size - ', len(train['features'][i]))
    tempTrainFeature = scipy.misc.imresize(train['features'][i], image_size_tuple , interp='bilinear')
    
    
    trainVertices = np.array([[(train['coords'][i][0],train['coords'][i][1]),
                               (train['coords'][i][0],train['coords'][i][3]), 
                               (train['coords'][i][2],train['coords'][i][3]), 
                               (train['coords'][i][2],train['coords'][i][1])]],
                            dtype=np.int32)
    tempTrainFeature=region_of_interest(tempTrainFeature, trainVertices)
    tempTrainFeature=tempTrainFeature.reshape(image_size_tuple)
    #tempTrainFeature=normalize_greyscale(tempTrainFeature)
    #feature = np.array(tempTrainFeature, dtype=np.float32).flatten()
    tempTrainFeatureResize = tempTrainFeature.reshape(image_size_flat)
    
    #print(len(feature))
    train["input-features"].append(tempTrainFeatureResize)     
    train["input-features-noresize"].append(tempTrainFeature)   
    #print('done - ', i)
for j, option in enumerate(test['features']):
    #print('Size - ', test['sizes'][j])
    tempTestFeature = scipy.misc.imresize(test['features'][j], image_size_tuple, interp='bilinear')
    
    testVertices = np.array([[(test['coords'][j][0],test['coords'][j][1]),
                               (test['coords'][j][0],test['coords'][j][3]), 
                               (test['coords'][j][2],test['coords'][j][3]), 
                               (test['coords'][j][2],test['coords'][j][1])]],
                            dtype=np.int32)
    tempTestFeature=region_of_interest(tempTestFeature, testVertices)
    tempTestFeature=tempTestFeature.reshape(image_size_tuple)
    #testfeature = np.array(tempTestFeature, dtype=np.float32).flatten()
    #tempTestFeature=normalize_greyscale(tempTestFeature)
    tempTestFeatureResize = tempTestFeature.reshape(image_size_flat)
    test["input-features"].append(tempTestFeatureResize)
    test["input-features-noresize"].append(tempTestFeature)  
    #print('done - ', j)

print("Train Feature Count = ", len(train["input-features-noresize"]))
print("Test Feature Count = ", len(test["input-features-noresize"]))
print('Pre processing done.')

def plot_images(images, labels, cls_pred=None, rect_coords=None, reshape=True):
    assert len(images) == len(labels) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        # Plot image.
        #newImgArr = scipy.misc.imresize(images[i], (30,30), interp='bilinear')
        if reshape==False:
            ax.imshow(images[i], cmap='binary') #.reshape(img_shape)
        else:
            ax.imshow(images[i].reshape(image_size_tuple), cmap='binary')
            
        # Create a Rectangle for the co-ordinates
        if rect_coords is not None:
            rect = patches.Rectangle((rect_coords[i][0],rect_coords[i][1]),(rect_coords[i][2]-rect_coords[i][0]),(rect_coords[i][3]-rect_coords[i][1]),linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(labels[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
# Get the images from the test-set.
#[38000:38009]
images = train['input-features-noresize'][0:9]

# Get the true labels for those images.
labels = train['labels'][0:9]

# Get the sizes for those images.
sizes = train['sizes'][0:9]
coords = train['coords'][0:9]
# Plot the images and labels using our helper-function above.
print("Original Images -")
plot_images(images=train['features'][0:9], labels=labels, cls_pred=None, rect_coords=coords, reshape=False)

print("New Reshaped Images -")
plot_images(images=images, labels=labels, cls_pred=None, rect_coords=coords, reshape=False)

# Plot the TEST images and labels using our helper-function above.
print("Original Test Images -")
plot_images(images=test['features'][0:9], labels=test['labels'][0:9], cls_pred=None, rect_coords=test['coords'][0:9], reshape=False)

print("New Reshaped Test Images -")
plot_images(images=test['input-features-noresize'][0:9], labels=test['labels'][0:9], cls_pred=None, rect_coords=test['coords'][0:9], reshape=False)

#img_size_flat = img_size * img_size
img_size_flat = 32 * 32 *3
num_classes = n_classes 
print(num_classes)

## COVNET NN GRAPH #####

# Parameters
learning_rate = 0.001
batch_size = 100
training_epochs = 30

layer_width = {
    'layer_1': 32,
    'layer_2': 64,
    'layer_3': 128,
    'fully_connected': 512
}

weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, 3, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'layer_3': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_2'], layer_width['layer_3']])),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [1024, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], n_classes]))
}
biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(n_classes))
}

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

#Create model
def conv_net(x, weights, biases):
    # Layer 1
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1, k=2)
    print("conv1=============", conv1.get_shape())
    
    # Layer 2
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2, k=2)
    print("conv2=============", conv2.get_shape())
    
    # Layer 3
    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv3 = maxpool2d(conv2, k=2)
    print("conv3=============", conv3.get_shape())
    
    # Fully connected layer
    ## Reshape conv3 output to fit fully connected layer input
    #fc1 = tf.reshape(
    #    conv2,
    #    [-1, weights['fully_connected'].get_shape().as_list()[0]])
    
    ### Flattening instead of reshape
    fc1 = tf.contrib.layers.flatten(conv3)
    print("fc1==FLAT===========", fc1.get_shape())
    
    fc1 = tf.add(
        tf.matmul(fc1, weights['fully_connected']),
        biases['fully_connected'])
    fc1 = tf.nn.relu(fc1)
    print("fc1=============", fc1.get_shape())
    
    # Output Layer - class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# tf Graph input
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

logits = conv_net(x, weights, biases)

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

session = tf.Session()
session.run(tf.initialize_all_variables())
batch_size = 100
batch_count = int(math.ceil(len(train['input-features'])/batch_size))

# Measurements use for graphing loss and accuracy
log_batch_step = 400
batches = []
loss_batch = []
train_acc_batch = []
train_full_acc_batch = []
valid_acc_batch = []

feed_dict_train_full = {x: train["input-features-noresize"],
                        y: train["hot-labels"],
                        y_true: train["hot-labels"],
                        y_true_cls: train["labels"],
                        }

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def optimize(num_iterations):
    for i in range(num_iterations):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Iterations {:>2}/{}'.format(i+1, num_iterations), unit='batches')
        
        for batch_i in batches_pbar:
            # Get a batch of training examples.
            
            batch_start = batch_i*batch_size
            #print("batch_i=", batch_i, ",batch_start=", batch_start, "batch_size=", batch_size)
            #print("train Length", len(train["input-features-noresize"]), ", " , len(train["hot-labels"]), ", " , len(train["labels"]))
            batch_features = train["input-features-noresize"][batch_start:batch_start + batch_size]
            batch_labels = train["hot-labels"][batch_start:batch_start + batch_size]
            batch_labels_cls = train["labels"][batch_start:batch_start + batch_size]
            
            feed_dict_train = {x: batch_features,
                               y: batch_labels,
                               y_true: batch_labels,
                               y_true_cls: batch_labels_cls,
                               }
            
            #print("x=", len(feed_dict_train[x]))
            #print("y=", len(feed_dict_train[y]))
            #print("y_true=", len(feed_dict_train[y_true]))
            #print("y_true_cls=", len(feed_dict_train[y_true_cls]))

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            #l = session.run([cost],feed_dict=feed_dict_train)
            #opt = session.run([optimizer],feed_dict=feed_dict_train)
            
            _, l = session.run([optimizer, cost],feed_dict=feed_dict_train)
            
            #print(batch_i % log_batch_step,", batches_pbar=",batches_pbar, ", batch_count=",batch_count, ",batch_start=", batch_start)
            
            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                #training_accuracy_local = session.run(accuracy, feed_dict=feed_dict_train)
                #training_accuracy_full = session.run(accuracy, feed_dict=feed_dict_train_full)
                #validation_accuracy = session.run(accuracy, feed_dict=feed_dict_test)
                # Log batches
                #previous_batch = batches[-1] if batches else 0
                #batches.append(log_batch_step + previous_batch)
                #print("batch_i=", str(batch_i) ,  "training_accuracy_local=", str(training_accuracy_local),
                #      ",Loss=" , str(l), ",training_accuracy_full=", str(training_accuracy_full),
                #      ",validation_accuracy=", str(validation_accuracy), "batches Length=", str(len(batches)))
                
                loss_batch.append(l)
                #train_acc_batch.append(training_accuracy_local)
                #train_full_acc_batch.append(training_accuracy_full)
                #valid_acc_batch.append(validation_accuracy)
        
        # Check accuracy against Validation data
        #validation_accuracy = session.run(accuracy, feed_dict=feed_dict_test)
        #print("Final Optimized Validation Accuracy = ", str(validation_accuracy))

feed_dict_test = {x: test["input-features-noresize"],
                  y: test["hot-labels"],
                  y_true: test["hot-labels"],
                  y_true_cls: test["labels"],
                  }
print(len(feed_dict_test[x]))
print(len(feed_dict_test[y]))
print(len(feed_dict_test[y_true]))
print(len(feed_dict_test[y_true_cls]))

def plot_accuracy():
    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches, loss_batch, 'g')
    #loss_plot.set_xlim([0, len(batches)])
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, train_full_acc_batch, 'g', label='Training Accuracy Full')
    acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
    
    acc_plot.set_ylim([0, 1.0])
    #acc_plot.set_xlim([batches[0], batches[-1]])
    #acc_plot.set_xlim([0, len(batches)])
    acc_plot.legend(loc=2)
    plt.tight_layout()
    plt.show()
    
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))
    
def print_weights(weightTensor):
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weightTensor)
    print("Total Weights = ", sum(w))
    #for i in range(num_classes):
    #    weights_x = w[:, i]
    #    print(i, "-->", sum(weights_x))

def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = feed_dict_test[y_true_cls]
    
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)
    
    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = np.array(feed_dict_test[x])[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = feed_dict_test[y_true_cls][incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9], labels=cls_true[0:9], cls_pred=cls_pred[0:9])
    #plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

def plot_example_success():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = np.array(feed_dict_test[x])[correct]
    images_len = len(images)
    # Get the predicted classes for those images.
    cls_pred = cls_pred[correct]

    # Get the true classes for those images.
    cls_true = feed_dict_test[y_true_cls][correct]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9], labels=cls_true[0:9], cls_pred=cls_pred[0:9])
    plot_images(images=images[images_len-9:images_len], 
                labels=cls_true[images_len-9:images_len], 
                cls_pred=cls_pred[images_len-9:images_len])

def plot_weights(weightTensor):
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weightTensor)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<num_classes:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, n_classes)
            image = w[:, i].reshape(32,16,1)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

print_accuracy()
plot_accuracy()
print("Plot Batches Length = ", len(batches))

print_weights(weights['out'])
optimize(num_iterations=1)
print_accuracy()
print_weights(weights['out'])