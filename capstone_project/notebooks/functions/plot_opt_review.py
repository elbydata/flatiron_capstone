import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os

def plot_opt_review(models_hists,
                    baseline=None,
                    prev_best=None,
                    trial_no=None,
                    roll=None,
                    val_acc_ylim=None,
                    val_loss_ylim=None):

    """
    Takes a list of models and histories and plots the training/validation accuracy and loss over the training epochs.
    
    Paramters:
        models_hists: (list) list of pairs of model names (string) and training history (dict - as returned by the 
                    keras model.fit function)
        baseline: (string with quotes) name of baseline model (or any other) as saved in directory (optional)
        prev_best: (string with quotes) name of any other model to compare as saved in directory (optional)
        trial_no: (int) experiment/trial number for labelling purposes
        roll: (int) includes rolling mean of validation data in visualization with window size given as 
        argument (defaults to None)
        val_acc_ylim: (tuple) sets y-axis range to tuple specified (min, max) on the accuracy graph
        val_loss_ylim: (tuple) sets y-axis range to tuple specified (min, max) on the loss graph
        
    Returns:
        Visualization of training and validation accuracy and loss over the training epochs
    """   
    
    font_dict = {'fontsize': 15}
    color_list = ['deepskyblue','salmon','gold','forestgreen','mediumorchid','darkmagenta','turquoise']  
              
    fig, axs = plt.subplots(3, 2, figsize=(22,15))

    if baseline:            
        with open(f'../notebooks/model_construction/saved_models/{baseline}_history', 'rb') as file:
            base_hist=pickle.load(file)            
        axs[0][0].plot(base_hist['acc'], color='gray', label='Baseline Train', linestyle=':')
        axs[0][1].plot(base_hist['loss'], color='gray', label='Baseline Train', linestyle=':')
        axs[2][0].plot([i-j for i,j in zip(base_hist['acc'],base_hist['val_acc'])], color='gray', label='Baseline TVA Dif')
        axs[2][1].plot([j-i for i,j in zip(base_hist['loss'],base_hist['val_loss'])], color='gray', label='Baseline TVA Dif')

        if roll:
            axs[1][0].plot(pd.DataFrame(base_hist['val_acc']).rolling(roll, min_periods=1).mean(),
                     color='gray', label='Baseline Val')
            axs[1][1].plot(pd.DataFrame(base_hist['val_loss']).rolling(roll, min_periods=1).mean(),
                     color='gray', label='Baseline Val')                

        else:            
            axs[1][0].plot(base_hist['val_acc'], color='gray', label='Baseline Val')
            axs[1][1].plot(base_hist['val_loss'], color='gray', label='Baseline Val')

    else:
        pass

    if prev_best:            
        with open(f'../notebooks/model_construction/saved_models/{prev_best}_history', 'rb') as file:
            prev_hist=pickle.load(file)                        
        axs[0][0].plot(prev_hist['acc'], color='black', label=f'{prev_best} Train', linestyle=':')
        axs[0][1].plot(prev_hist['loss'], color='black', label=f'{prev_best} Train', linestyle=':')
        axs[2][0].plot([i-j for i,j in zip(prev_hist['acc'],prev_hist['val_acc'])],color='black',
                       label=f'{prev_best} TVA Dif')
        axs[2][1].plot([j-i for i,j in zip(prev_hist['loss'],prev_hist['val_loss'])],color='black',
                       label=f'{prev_best} TVA Dif')

        if roll:
            axs[1][0].plot(pd.DataFrame(prev_hist['val_acc']).rolling(roll, min_periods=1).mean(),
                     color='black', label=f'{prev_best} Val')
            axs[1][1].plot(pd.DataFrame(prev_hist['val_loss']).rolling(roll, min_periods=1).mean(),
                     color='black', label=f'{prev_best} Val')                

        else:                
            axs[1][0].plot(prev_hist['val_acc'], color='black', label=f'{prev_best} Val')
            axs[1][1].plot(prev_hist['val_loss'], color='black', label=f'{prev_best} Val')

    else:
        pass

    for h in models_hists:
        axs[0][0].plot(h[1]['acc'], color=color_list[models_hists.index(h)], label=f'{h[0]} Train', linestyle=':')
        axs[0][1].plot(h[1]['loss'], color=color_list[models_hists.index(h)], label=f'{h[0]} Train', linestyle=':')
        axs[2][0].plot([i-j for i,j in zip(h[1]['acc'],h[1]['val_acc'])],
                       color=color_list[models_hists.index(h)], label=f'{h[0]} TVA Dif')
        axs[2][1].plot([j-i for i,j in zip(h[1]['loss'],h[1]['val_loss'])],
                       color=color_list[models_hists.index(h)], label=f'{h[0]} TVA Dif')

        if roll:
            axs[1][0].plot(pd.DataFrame(h[1]['val_acc']).rolling(roll, min_periods=1).mean(),
                     color=color_list[models_hists.index(h)], label=f'{h[0]} Val')
            axs[1][1].plot(pd.DataFrame(h[1]['val_loss']).rolling(roll, min_periods=1).mean(),
                     color=color_list[models_hists.index(h)], label=f'{h[0]} Val')                

        else:
            axs[1][0].plot(h[1]['val_acc'], color=color_list[models_hists.index(h)], label=f'{h[0]} Val')
            axs[1][1].plot(h[1]['val_loss'], color=color_list[models_hists.index(h)], label=f'{h[0]} Val')

    axs[0][0].set_title('Training Accuracy', fontdict=font_dict)
    axs[0][0].set_xlabel('Epoch')
    axs[0][0].legend(loc='best')

    axs[0][1].set_title('Training Loss', fontdict=font_dict)
    axs[0][1].set_xlabel('Epoch')
    axs[0][1].legend(loc='best')

    axs[1][0].set_title('Validation Accuracy', fontdict=font_dict)
    axs[1][0].set_xlabel('Epoch')
    axs[1][0].legend(loc='best')
    if val_acc_ylim:
        axs[1][0].set_ylim(val_acc_ylim)
    else:
        pass 

    axs[1][1].set_title('Validation Loss', fontdict=font_dict)
    axs[1][1].set_xlabel('Epoch')
    axs[1][1].legend(loc='best')
    if val_loss_ylim:
        axs[1][1].set_ylim(val_loss_ylim)
    else:
        pass 

    axs[2][0].set_title('Training & Validation Accuracy Difference', fontdict=font_dict)
    axs[2][0].set_xlabel('Epoch')
    axs[2][0].legend(loc='best')
    axs[2][0].axhspan(0, 0.2, color='deepskyblue', alpha=0.2)

    axs[2][1].set_title('Training & Validation Loss Difference', fontdict=font_dict)
    axs[2][1].set_xlabel('Epoch')
    axs[2][1].legend(loc='best')      

    if trial_no:
        fig.suptitle(f'Optimization Trials for Model {trial_no}', fontsize=20)
    else:
        fig.suptitle('Optimization Trials', fontsize=20)        
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.975])
    plt.show()
    