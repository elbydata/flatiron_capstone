import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def test_inspect(y_pred,
                 y_test,
                 model_name=None):
    
    """
    Takes the images from the model test datasets and creates a visualization of the predicted versus the actual instances
    of each class.
    
    Paramters:
        y_pred: (array) the predicted labels for the test set
        y_test: (array) the actual labels for the test set (one-hot encoded)
        model_name: (string with quotes) model name for labelling purposes
        
    Returns:
        Bar chart displaying the number of predicted and atual instances of each class in the testing set.
    """  
    
    if model_name:
        
        species_df = pd.read_csv('../data/0003_general/species_list_final.csv')
        species_as_list = species_df['common_name'].tolist()
        
        y_actual = pd.DataFrame(y_test).idxmax(axis=1).to_list()
        test_preds = y_pred
        
        named_test_actuals = []
        named_test_preds = []
        
        for i in y_actual:
            named_test_actuals.append(species_as_list[i])
        
        for i in test_preds:
            named_test_preds.append(species_as_list[i])

        df_pred_count = pd.DataFrame.from_dict(
            Counter(named_test_preds), orient='index').reset_index().rename(columns={'index':'species', 0:'predicted'})
        
        df_act_count = pd.DataFrame.from_dict(
            Counter(named_test_actuals), orient='index').reset_index().rename(columns={'index':'species', 0:'actual'})
        
        big_df = df_act_count.merge(df_pred_count, how='left')
        
        melt_df = pd.melt(big_df, id_vars=['species'], value_vars=['predicted', 'actual']).sort_values('species')
        
        sns.set_style(style='darkgrid')
        plt.figure(figsize=(12,8))
        sns.barplot(x='value', y='species', hue='variable', data=melt_df, palette=['deepskyblue','salmon'])
        plt.title(f'{model_name}: Predicted Vs. Actual', fontdict={'fontsize':15})
        plt.xlabel('Number')
        plt.ylabel('Species')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        

    else:
        print('Please specify model name')
    