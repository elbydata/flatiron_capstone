import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models

def identify_mushroom(user_img):
    
    """
    Runs the mushroom identification model on an image in the repository
    
    Paramters:
        model_name: (string with quotes) relative path to the image
        
    Returns:
        Visualization of the image, along with a classification result
    """            
    
    species_list = ['Amethyst Deceiver',
                'Bolete',
                'Chanterelle',
                'Chicken Of The Woods',
                'Death Cap',
                'False Chanterelle',
                'False Morel',
                'Fibrecap',
                'Field Mushroom',
                'Fly Agaric',
                'Giant Puffball',
                'Grey Oyster',
                'Morel',
                'Orange Peel',
                'Roundhead',
                'Saddle',
                'Shaggy Inkcap',
                'Stinkhorn',
                'Waxcap',
                'Yellow Stainer']

    model_x = models.load_model(f'../notebooks/model_construction/saved_models/model_13.h5')
    
    base_img = cv2.imread(user_img)
    img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    resized_img = cv2.resize(img,(200,200),interpolation=cv2.INTER_CUBIC)  
    expand_norm_img = (np.expand_dims(resized_img, axis=0))*(1./255)

    layer_outputs = [layer.output for layer in model_x.layers] 
    activation_model = models.Model(inputs=model_x.input, outputs=layer_outputs)
    activations = activation_model.predict(expand_norm_img)
    
    prob_list = list(np.round((activations[-1][0])*100,2))
    top_result = species_list[prob_list.index(max(prob_list))]   
    
    fig, axs = plt.subplots(2,1, figsize=(5,9))
    fig.suptitle(f'Most probably: {top_result}', fontsize=20, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    axs[0].imshow(img)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    axs[1].barh(np.arange(len(species_list)), prob_list)
    for i, v in enumerate(prob_list):
        if v == max(prob_list):
            axs[1].text(v+3, i+.25, str(v)+'%', color='blue', fontweight='bold')
        else:
            axs[1].text(v+3, i+.25, str(v)+'%', color='blue')
    axs[1].set_title('Probability breakdown:', fontweight='bold', loc='left')
    axs[1].tick_params(axis=u'both', which=u'both',length=0)
    axs[1].set_xticks([])
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    
    plt.yticks(np.arange(len(species_list)), species_list)
    plt.gca().invert_yaxis()
    