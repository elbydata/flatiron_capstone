![cover_pic](mushroom_identifier/data/0003_general/cover_pic.jpg)

# Mushroom Identifier

Using a convolutional neural network to recognise and classify common British mushrooms into species.


### Project Summary

There are roughly 4'500 species of mushrooms growing in the UK, some of which are fatally poisonous, and many of which are deceptively similar in appearance. Foragers and nature enthusiasts who pick wild mushrooms, therefore, run the risk of misidentifying mushroom species. This can be a deadly mistake.

This project aims to provide a means of determining the species of a mushroom based on an image, with the scope limited to the top 20 most common UK species. The goal was to construct a model that achieved a test accuracy greater than 80% (compared to a 5% chance).

The models were trained and validated using a set of approximately 10'000 images obtained from a combination of sources including a competition run by the Atlas of Danish Fungi (see Data Source 1), and an image database maintained by mushroomobserver.org (Data Source 2). The baseline model achieved a classification accuracy of 43.39% on the test data over the 20 classes. The model was then improved through adjustment of the network architecture and hyperparameters, and through the use of data augmentation. Over 13 model iterations, the winning model achieved an accuracy of 81.18% on the test data.


### Data

The models were constructed using a dataset that included approximately 10'000 images (see sources and acknowledgements below) covering 20 different mushroom species common to the United Kingdom. The data is therefore split into 20 classes, and the task of the models is to classify each image in the testing set to a high enough degree of accuracy. Of the 20 species, approximately half are poisonous. 

Subsequently to this project, future work could include the sourcing of additional images to improve the model's accuracy. Furthermore, the list of classes could be extended to include more species, and potentially expand to cover mushroom species outside of the UK and Europe.


### Contact Info

For any queries or additional information, please email info@elbydata.com


### Sources and Acknowledgements

The image data for the project was obtained from the following sources:
- **Data Source 1:** Atlas of Danish Fungi competition data (training set only): https://github.com/visipedia/fgvcx_fungi_comp/blob/master/README.md, see also https://svampe.databasen.org
- **Data Source 2:** Mushroom Observer database https://mushroomobserver.org (see also specific csv files hosted by the website under: data/0003_mo_scrape/mo_csv_files/ downloadable at https://mushroomobserver.org/'database_name'.csv)

Special thanks to:
- Jason at mushroomobserver.org for providing support with the image scraping
- Poppy at https://www.wildfooduk.com for kindly allowing use of their images


##

<p align="center">
    <img src="mushroom_identifier/data/0003_general/disclaimer_pic.jpg" width="400" height="200" />
</p>    

### Disclaimer

This project was created for academic purposes only and is not intended for commercial or recreational use. The classification results are not, and do not purport to be, adequate for identification of mushroom species. Wild mushrooms can be highly toxic and potentially lethal. The project creator makes no representations or warranties relating to any of the data used or any consequences associated with any classification result.

The Danish Svampe Atlas makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.

### License

The project is licensed under the MIT license.
