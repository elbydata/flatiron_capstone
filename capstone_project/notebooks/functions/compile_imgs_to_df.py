import pandas as pd
import numpy as np
import os

def compile_imgs_to_df(class_name,
                       base_path):

    """
    Creates an x-by-2 dataframe of named and classified images for a particular class. The function assumes that images 
    are stored in subdirectories for each class, ie: ../base_path/class_name/.
    
    Paramters:
        class_name: the name of the class as appears in the subdirectory containing images of that class
        base_path: the base path to the directory containing all of the class subdirectories
        
    Returns:
        df: an x-by-2 dataframe of names and classified images for the specified class
    """
    
    img_list = []

    for i in os.listdir(f'{base_path}{class_name}'):
        img = f'{base_path}{class_name}/'+i
        img_list.append(img)
    
    df = pd.DataFrame(img_list, columns=['image'])
    df['class'] = class_name
    
    return df
