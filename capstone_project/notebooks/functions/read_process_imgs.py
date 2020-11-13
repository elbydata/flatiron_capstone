import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def read_process_imgs(image_df,
                      x_size,
                      y_size):
    
    """
    Takes an x-by-2 dataframe of named and classified images (where the first column is the image name and the second column
    is the image class) and processes it into a set of numpy arrays for use in a neural network. Reading and resizing is
    performed by the cv2 library, and class values are one-hot encoded using the relevant sklearn.preprocessing and keras.utils 
    libraries.
    
    Paramters:
        image_df: (dataframe) the x-by-2 dataframe of named and classified images
        x_size: (integer) the disired horizontal output size of each image in pixels
        y_size: (integer) the disired vertical output size of each image in pixels
        
    Returns:
        X: the numpy array form of the image data; shape: (number of images, x_size, y_size, 3)
        y: the numpy array form of the class data; shape: (number of images, number of classes)
        y_list: a list of class values in string form for each image in X (used primarily for labeling visualisations)
    """
    
    X_list = []
    y_list = []
    xs = int(x_size)
    ys = int(y_size)
    
    shuffle_df = image_df.sample(frac=1).reset_index(drop=True)
    
    for index, row in shuffle_df.iterrows():
        raw_img = cv2.imread(row[0],1)
        try:
            resized_img = cv2.resize(raw_img, (xs,ys),interpolation=cv2.INTER_CUBIC)
        except cv2.error as e:
            pass
        
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        X_list.append(rgb_img)
        y_list.append(row[1])
        
    X = np.array(X_list)
    
    encoder = LabelEncoder()
    encoder.fit(y_list)
    encoded_y = encoder.transform(y_list)
    y = np_utils.to_categorical(encoded_y)
    
    return X, y, y_list