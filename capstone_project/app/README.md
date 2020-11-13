![cover_pic](cover_pic.jpg)

# Mushroom Identifier - App


This is the app version of the Mushroom Identifier; a convolutional neural network which aims recognise and classify common British mushrooms into species.

The mothership for this app can be found at https://github.com/elbydata/mushroom_identifier. The repository contains the the entire project, including the image data sets and all of the trained models. This app deploys the best version of the model, which achieved an accuracy of 81.18% on the testing data. For more information, including all of the source code for the model, please refer to the linked repo.

### CURRENT STATUS

The app has not yet been deployed with Heroku because of an issue loading the .h5 model.


### Repository Navigation

This repo contains the following:

 * **.gitattributes**: large file storage extension instructions and other attributes
 * **cover_pic.jpg**: mushroom collage found at the top of this README
 * **mi_13.h5**: the trained mushroom identification model (in Keras .h5 format)
 * **mi_app.py** the source code for the app (built through Streamlit)
 * **Procfile** Procfile for deployment through Heroku
 * **README.md**: the document you are currently reading!
 * **requirements.txt** environment package requirements
 * **setup.sh** setup instructions for Streamlit/Heroku


### Contact Info

For any queries or additional information, please email info@elbydata.com


### Disclaimer

This project was created for academic purposes only and is not intended for commercial or recreational use. The classification results are not, and do not purport to be, adequate for identification of mushroom species. Wild mushrooms can be highly toxic and potentially lethal. The project creator makes no representations or warranties relating to any of the data used or any consequences associated with any classification result.

The Danish Svampe Atlas makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
