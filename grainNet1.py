from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import custom_data_loader

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    '''model function for CNN'''
    # input layer
    input_layer = tf.reshape(features['x'], [1,28,28,3])# [batch_size, image_height, image_width, channels]
    # this will need to be changed for final application
    # batch size = number of training examples
    # image height, width = pixel value of image size (28X28)
    # channels = number of color values 3=rgb or 1=monochromatic (black or white only)
    # MNIST channels = 1 bc its black-white only
    # COULD POTENTIALLY SCALE TO REDUCE TO BLACKWHITE ONLY AND KEEP CHANNELS AS 1 BUT OTHERWISE WOULD REQUIRE 3
    # batch size set to -1 so that it is dynamically scaled based off of features['x'] size

    # convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32, # number of filters applied
                             kernel_size=[5,5], # filter size
                             padding='same', # defines output tensor to have same size as input tensor
                             activation=tf.nn.relu)# connected to input layer of shape [batch size, 28, 28, 1(channels)]
                                # output of this will have 32 channels instead of 1, each holding the output from each filter

    # pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2,2], # extract 2X2 subregions
                                    strides=2) # separate subimages by 2 pixels
                                        # output of this will be [batch_size, 14, 14, 32] as the pool decreases size by 2


    # convolutional layer 2 and pooling layer 2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,# 64 filters of size (5X5) as seen below
                             kernel_size=[5,5],
                             padding='same',
                             activation=tf.nn.relu)
                            # now will have shape [batch_size, 14, 14, 64] 64 filters applied

    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2,2],# 2X2 subregions
                                    strides=2)# separated by 2 pixels
                                    # now will have shape [batch_size, 7, 7, 64] as it has been reduced by factor 2 again

    #dense layer
    pool2_flat = tf.reshape(pool2, [1,7*7*64])# again, batch size is dynamically calculated using -1
                                                # this flattens the tensor to only 2 dimensions size [batch_size, 3136]
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024,# number of neurons in dense layer
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,# helps improve the recognition capabilities by dropping 40% of the elements
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)# output has size [batch_size, 1024]

    #logits layer
    logits = tf.layers.dense(inputs=dropout, units=764)################ MUST BE CHANGED TO 28*28=764 FOR FINAL APPLCATION
    # final size is [batch_size, 10] we want [batch_size, 764]

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # add softmax_tensor to the graph, used in PREDICT and by logging_hook
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # calculate loss (for TRAIN and EVAL mode)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) #cross entropy is used as the loss function here
    ############# LABELS IS A LIST OF PREDICTION INDICES FOR THE EXAMPLES, FOR MNIST I.E. [1,9,2,3,...]
    ############# FOR FINAL APPLICATION IT WOULD BE [[BLACK-WHITE IMAGE],[BLACK-WHITE IMAGE],[BLACK-WHITE IMAGE],...]

    # configure training operation (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)######## MAY NEED TO BE SLOWED DOWN FOR FINAL APPLICATION
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])}# adds an accuracy metric to the model
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# loading training and test data
def main(unused_argv):
    # load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("") ########### CHANGE TO OWN CUSTOM DATASET
    ####### ALSO CHANGE NAME OF THE MODEL FROM MNIST TO SOMETHING ELSE
    train_data = mnist.train.images # returns np.array ## store training data raw pixel values of 55k images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32) # stores labels [1,2,9,5,...] for all images
    eval_data = mnist.test.images # returns np.array ## stores the raw pixel value for all images to be tested
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # create estimator, tensorflow class for performing high level model training and evaluation
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir='MNIST/mnist_convnet_model')

    # create logging hook to track progress while training
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=50) # logs probabilities from softmax_tensor every 50 iterations

    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, # calling training data
                                                         y=train_labels, # calling training labels
                                                         batch_size=100, # train with 100 images at a time
                                                         num_epochs=None, # train until specified step count is reached
                                                         shuffle=True) # shuffles training data to get a better result
    mnist_classifier.train(input_fn=train_input_fn,
                           steps=20000, # train for 20k steps
                           hooks=[logging_hook]) # ensures probabilities from softmax tensor is displayed while running

    # evaluate model once training is completed and print results
    # this is the part that actually matters the most
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                       y=eval_labels,
                                                       num_epochs=None,
                                                       shuffle=False)





# application logic stored here

if __name__ == "__main__":
    tf.app.run()