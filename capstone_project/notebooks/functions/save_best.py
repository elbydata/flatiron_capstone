from keras import models
import pickle
import os
from send2trash import send2trash

def save_best(name_list,
              best_history,
              best_name=None,
              save_as=None):
    
    """
    Saves the selected model from a list of models, and sends the others to the trash.
    
    Paramters:
        name_list: (list of strings) names of models from the trial
        best_history: (Keras history object) history dictionary for the model chosen to be saved
        best_name: (string with quotes) name of the chosen model from the trial
        save_as: (string with quotes) name under which the chosen model will be saved in the folder
        
    Returns:
        Message confirmation of execution
    """ 
    
    if os.path.exists(f'../notebooks/model_construction/saved_models/{best_name}.h5'):
        
        best_model = models.load_model(f'../notebooks/model_construction/saved_models/{best_name}.h5')
        best_model.save(f'../notebooks/model_construction/saved_models/{save_as}.h5')            

        print(f'{best_name} saved as {save_as} in ../notebooks/model_construction/saved_models/!')
        print('---')
        with open(f'../notebooks/model_construction/saved_models/{save_as}_history', 'wb') as file_pi:
            pickle.dump(best_history, file_pi)
        print(f'{best_name}_history saved as {save_as}_history saved in ../notebooks/model_construction/saved_models/!')
        print('---')

    else:
        print(f'Cannot save {best_name} because the file does not exist - maybe already saved as {save_as}')    

    name_list_2 = name_list

    for m in name_list_2:
        if os.path.exists(f'../notebooks/model_construction/saved_models/{m}.h5'):
            send2trash(f'../notebooks/model_construction/saved_models/{m}.h5')
            print(f'{m} moved to trash!')
        else:
            print(f'Cannot remove {m} because the file does not exist - maybe already removed')
    
    