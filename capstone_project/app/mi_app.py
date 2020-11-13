import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from keras import models
from PIL import Image, ImageOps

def main():

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.write("""# Mushroom Identifier""")

    st.write('This is a simple image classification web app to predict mushroom species')
    file = st.file_uploader('Please upload an image file and scroll down for results!', type=['jpg','jpeg','png'])

    def identify_mushroom(user_img):       

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

        model_x = models.load_model('mi_13.h5')

        size = (200,200)    
        img_read = ImageOps.fit(user_img, size, Image.ANTIALIAS)
        img_array = np.asarray(img_read)

        expand_norm_img = (np.expand_dims(img_array, axis=0))*(1./255)

        layer_outputs = [layer.output for layer in model_x.layers] 
        activation_model = models.Model(inputs=model_x.input, outputs=layer_outputs)
        activations = activation_model.predict(expand_norm_img)

        prob_list = list(np.round((activations[-1][0])*100,2))
        top_result = species_list[prob_list.index(max(prob_list))]  

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        fig.suptitle(f'Most probably: {top_result}', fontsize=20, fontweight='bold')

        ax.barh(np.arange(len(species_list)), prob_list)
        for i, v in enumerate(prob_list):
            if v == max(prob_list):
                ax.text(v+3, i+.25, str(v)+'%', color='blue', fontweight='bold')
            else:
                ax.text(v+3, i+.25, str(v)+'%', color='blue')
        ax.set_title('Probability breakdown:', fontweight='bold', loc='left')
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.yticks(np.arange(len(species_list)), species_list)
        plt.gca().invert_yaxis()

    if file is None:
        pass
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = identify_mushroom(image)
        st.write(prediction)

    st.pyplot()    
    
if __name__ == '__main__':
    main()
