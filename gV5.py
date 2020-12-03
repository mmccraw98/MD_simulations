from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from statistics import mean
import tensorflow as tf
#from functions import saveData, getNumberFiles
import os, os.path
import random

from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from keras.models import Sequential


#### GV4 IS A TENSORFLOW CNN THAT DETECTS DEGREE OF A LINE OF BEST FIT FOR THE GRAIN BOUNDARY
#### IT DOES NOT PREDICT WHERE THE GRAIN BOUNDARY WOULD LIE OR IF IT EXISTS IN THE PICTURE

def create_comparison_plots(image1,image2,stacked=True):
    fig = plt.figure(figsize=(10, 5))
    columns = 2
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image1)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(image2)
    if stacked!=True:
        plt.show()
def getNumberFiles(path):
    '''
    must define path internally, assumes training input and output have same amount of training data
    '''
    DIR = path
    return (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
def deconstruct_image(array):
    new = []
    for row in array:
        for pixel in row:
            new.append(mean(pixel[:3]))
    return np.array(new)
def reconstruct_image(vector,ORIGINAL_IMAGE_SHAPE):
    count = 0
    new = np.zeros(ORIGINAL_IMAGE_SHAPE)
    for row in range(ORIGINAL_IMAGE_SHAPE[0]):
        for pixel in range(ORIGINAL_IMAGE_SHAPE[1]):
            new[row][pixel][0] = \
                new[row][pixel][1] = \
                new[row][pixel][2] = \
                vector[count]
            count += 1
    return new
def read_text_to_array(path):
    file = open(path, "r")
    array = np.array([float(line) for line in file.readlines()])
    file.close()
    return array

FILE_PATH = 'C:/Users/17576/PycharmProjects/imPort/images/grains/Training Data/'
ORIGINAL_IMAGE_SHAPE = (100, 100, 3)
NUM_FILES = getNumberFiles(FILE_PATH+'Training Input/')
TRAINING_IMAGES_INPUT = np.zeros((NUM_FILES,ORIGINAL_IMAGE_SHAPE[0],ORIGINAL_IMAGE_SHAPE[1]))
TRAINING_IMAGES_OUTPUT = np.zeros((NUM_FILES,ORIGINAL_IMAGE_SHAPE[0],ORIGINAL_IMAGE_SHAPE[1]))


print('------LOADING TRAINING INPUT IMAGES------')
for file_number in range(1,NUM_FILES+1):
    # load specified file
    print('LOADING IMAGE:'+str(file_number))
    image = deconstruct_image(np.array(Image.open(FILE_PATH + 'Training Input/' + 'Grain_Input_' + str(file_number) + '.png')))
    new = image.reshape((ORIGINAL_IMAGE_SHAPE[0],ORIGINAL_IMAGE_SHAPE[1]))
    for row in range(len(new)):
        for pixel in range(row):
            TRAINING_IMAGES_INPUT[file_number-1][row][pixel] = new[row][pixel]
print('------LOADING DONE------')

print('------LOADING TRAINING OUTPUT IMAGES------')
for file_number in range(1,NUM_FILES+1):
    print('LOADING IMAGE:'+str(file_number))
    image = read_text_to_array(FILE_PATH+'Training Output/Grain_Output_'
                               +str(file_number)).reshape((ORIGINAL_IMAGE_SHAPE[0],ORIGINAL_IMAGE_SHAPE[1]))
    for row in range(len(image)):
        for pixel in range(row):
            TRAINING_IMAGES_OUTPUT[file_number-1][row][pixel] = image[row][pixel]
print('------LOADING DONE------')

train_x = TRAINING_IMAGES_INPUT # probably not very good but re-using the training images as the testing images
test_x = TRAINING_IMAGES_INPUT





train_y = TRAINING_IMAGES_OUTPUT
test_y = TRAINING_IMAGES_OUTPUT
# Images fed into this model are 512 x 512 pixels with 3 channels
img_shape = (100,100,1)

# Set up model
model = Sequential()
model.add(Conv2D(32,5,input_shape=img_shape))
model.add(Activation('relu'))
model.add(MaxPool2D(2,strides=2))
model.add(Conv2D(64,5))
model.add(Activation('relu'))
model.add(MaxPool2D(2,strides=2))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10000)) # OUTPUT LAYER SIZE - 1
model.add(Activation('softmax'))
model.summary()

"""Before the training process, we have to put together a learning process 
in a particular form. It consists of 3 elements: an optimiser, a loss function and a metric."""
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
# dataset with handwritten digits to train the model on
from keras.datasets import mnist


print(train_x.shape)

train_x = np.expand_dims(train_x,-1)
test_x = np.expand_dims(test_x,-1)

# Train the model, iterating on the data in batches of 32 samples
# for 10 epochs
model.fit(train_x, train_y, batch_size=5, epochs=20, validation_data=(test_x,test_y))






newImage = deconstruct_image(np.array(Image.open(FILE_PATH+'Training Input/'+'Grain_Input_5.png')))
new = newImage.reshape((ORIGINAL_IMAGE_SHAPE[0], ORIGINAL_IMAGE_SHAPE[1]))

a = np.zeros((1,100,100,1))

for row in range(a.shape[1]):
    for pixel in range(a.shape[2]):
        (a[0][row][pixel]) = new[row][pixel]

PREDICT = model.predict_classes(a)
# show the inputs and predicted outputs
print(PREDICT[0])
