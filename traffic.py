import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

EPOCHS = 5
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.3

label_names = np.arange(0, 43)


def main():

    # Check command-line arguments

    #if len(sys.argv) not in [1, 2, 3]:
     #   sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data('/Users/suhaib/Documents/VSCode Projects/traffic/gtsrb')

    for idx in range(0, len(images)):
        images[idx] = images[idx] / 255.0


    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    print(x_train.shape)
    print(x_test.shape)

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    model.evaluate(x_test, y_test, verbose=2)

    return model, x_train, x_test, y_train, y_test


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    subfile_names_raw = os.listdir(data_dir)
    subfile_names = []
    for subfile in subfile_names_raw:
        if '.DS_Store' not in subfile:
            subfile_names.append(subfile)

    for subfile in subfile_names:
        subfolder_path = data_dir + '/' + subfile
        print(subfolder_path)
        image_files = os.listdir(subfolder_path)
        for image in image_files:
            image_path = subfolder_path + '/' + image
            #image_array = cv2.resize(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), (IMG_WIDTH, IMG_HEIGHT, 3))
            image_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            dim = (IMG_WIDTH, IMG_HEIGHT)
            image_resized = cv2.resize(image_raw, dim, interpolation = cv2.INTER_AREA)
            #image_array = np.ndarray((IMG_WIDTH, IMG_HEIGHT, 3))
            #for idx in range(0, 1, 2):
            #    img = image_raw[:, :, idx]
            #    image_array[:, :, idx] = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            images.append(image_resized)
            labels.append(int(subfile))
    return (images, labels)        

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    #Create neural network
    model = tf.keras.models.Sequential([

        #Convolutional layers
        tf.keras.layers.Conv2D(
            90, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.Conv2D(
            90, (5, 5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        #Max-pooling layer, using 12x12 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(12, 12)),

        #Flatten units
        tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        #Add a hidden layer with dropout
        tf.keras.layers.Dense(350, activation='relu'),

        tf.keras.layers.Dropout(0.5),

        #Add an output layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'),
    ])

    #Train neural network
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    main()
