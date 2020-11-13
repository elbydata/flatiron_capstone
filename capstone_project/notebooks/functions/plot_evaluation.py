import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def plot_evaluation(model_name,
                    roll=None):

    """
    Creates a plot of the training/validation accuracy and loss over the training epochs.
    
    Paramters:
        model_name: (string with quotes) name of model to evaluate (note: the model's history dictionary (as 
        returned by the keras model.fit function) is required to be in the 'saved models' folder
        roll: (int) includes rolling mean of validation data in visualization with window size given as 
        argument (defaults to None)
        
    Returns:
        Visualization of training and validation accuracy and loss over the training epochs
    """    
    
    with open(f'../notebooks/model_construction/saved_models/{model_name}_history', 'rb') as file:
            model_history=pickle.load(file)      
            
    fig, axs = plt.subplots(1, 2, figsize=(22,8))
    fig.suptitle(f'{model_name} Evaluation', fontsize=15)

    if roll:
        
        va_df = pd.DataFrame(model_history['val_acc'])
        vl_df = pd.DataFrame(model_history['val_loss'])
        va_rm = va_df.rolling(roll, min_periods=1).mean()
        vl_rm = vl_df.rolling(roll, min_periods=1).mean()          
        axs[0].plot(va_rm, color='deepskyblue', alpha=0.7, label='Validation Accuracy (Rolling Mean)')
        axs[1].plot(vl_rm, color='red', alpha=0.7, label='Validation Loss (Rolling Mean)')
        
    else:
        pass   
    
    axs[0].plot(model_history['acc'], color='deepskyblue', label='Training Accuracy', linestyle='--')
    axs[0].plot(model_history['val_acc'], color='deepskyblue', label='Validation Accuracy')
    axs[0].axhline(0.05, color='forestgreen', linestyle='dotted', label='Chance Line')
    axs[0].set_title('Training & Validation Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='best')    

    axs[1].plot(model_history['loss'], color='salmon', label='Training Loss', linestyle='--')
    axs[1].plot(model_history['val_loss'], color='salmon', label='Validation Loss')
    axs[1].set_title('Training & Validation Loss')
    axs[1].set_ylabel('Loss') 
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='best')

    plt.show()