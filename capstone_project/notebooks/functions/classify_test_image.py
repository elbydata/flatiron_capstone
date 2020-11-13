import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from keras import models


def classify_test_image(test_no,
                        model_name=None):

    """
    Takes an image from the model test dataset and runs a selected model classifier on it, outputting the model's classification 
    of the image, as well as a result (correct or incorrect) and displays the image.
    
    Paramters:
        test_no: (integer) the index number of the test image in the test set
        model_name: (string with quotes) the model to be used to make the classification
        
    Returns:
        Visualization of the image, along with a classification result (correct or incorrect)
    """    
    
    if model_name:
        
        species_df = pd.read_csv('../data/0003_general/species_list_final.csv')
        species_as_list = species_df['common_name'].tolist()
        X_test_non_norm = np.load('../data/0002_array_data/test_data/X_test_data.npy')
        X_test = X_test_non_norm*(1./255)
        y_test = np.load('../data/0002_array_data/test_data/y_test_data.npy')

        model = models.load_model(f'../notebooks/model_construction/saved_models/{model_name}.h5')
        test_img = X_test[test_no]
        test_img_rs = np.expand_dims(test_img, axis=0)
        y_act = np.where(y_test[test_no] == 1)[0][0]
        ypred = np.argmax(model.predict(test_img_rs),axis=-1)

        if int(y_act) == int(ypred[0]):
            result = 'Correct!'
        else:
            result = 'Incorrect'

        plt.figure(figsize=(7,7))
        plt.imshow(test_img)
        plt.title(f'{model_name} -- test no: {test_no}')
        plt.xticks([])
        plt.yticks([])
        print(f'Actual: {species_as_list[y_act]}')
        print(f'Predicted: {species_as_list[ypred[0]]}')
        print(f'Result: {result}')

    else:
        print('Please specify model name')
    