import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

def make_cm(y_pred,
            y_test,
            model_name=None):

    """
    Takes the images from the model test datasets and creates a confusion matrix for the classifications.
    
    Paramters:
        y_pred: (array) the predicted labels for the test set
        y_test: (array) the actual labels for the test set (one-hot encoded)
        model_name: (string with quotes) model name for labelling purposes
        
    Returns:
        Visualization of the confusion matrix for the model
    """    
    
    if model_name:
        
        species_df = pd.read_csv('../data/0003_general/species_list_final.csv')
        species_as_list = species_df['common_name'].tolist()
        
        test_preds = y_pred
        test_results_df = pd.DataFrame(test_preds,columns=['predicted'])
        test_results_df['actual'] = pd.DataFrame(y_test).idxmax(axis=1)
        
        cm = metrics.confusion_matrix(test_results_df['actual'],test_results_df['predicted'])
        cmn = np.true_divide(cm, cm.sum(axis=1, keepdims=True))*100
        df_cmn = pd.DataFrame(cmn, species_as_list, species_as_list)

        fig, ax1 = plt.subplots(1,1,figsize=(10,10))
        sns.heatmap(df_cmn, cmap='Blues', cbar=False, annot=True,annot_kws={"size":11}, fmt='.0f', linewidth=0.5)
        for t in ax1.texts:
            t.set_text(t.get_text()+'%')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {model_name}', fontdict={'fontsize':15})
        plt.tight_layout()
        plt.show()
        
    else:
        print('Please specify model name')
    