# Tidalsaurus: Detecting Galaxy Tidal Features with Machine Learning

This repository holds code and documentation for an LSSTC funded project as part of the 2021 Enabling Science Call, and part of the ISSC Ambassadors scheme aiming to connect members of different Science Collaborations within the Rubin Community.

Project lead: [Alice Desmons](https://github.com/a-desmons)

Supervisors: Sarah Brough, [Francois Lanusse](https://github.com/EiffL)

This project uses Nearest Neighbour Contrastive Learning Representation (NNCLR) methods to build a semi-supervised approach to the detection of tidal features. This repo contains the code used to create, train, validate and test the self-supervised machine learning model presented in Desmons et al. (in prep). 

The code runs on Python 3.7.4. Other versions of Python may work but have not been tested.

The requirements.txt file includes the python packages and versions used to run this code. Other versions of these packages may work but have not been tested.

## Model Summary
---
The model can be separated into two components: A self-supervised encoder, and a supervised finetuned classifier. 

The encoder uses image augmentations to learn to encode 128x128 pixel images into meaningful 128-dimensional representations. This part of the model is trained using unlabelled data composed of ~45,000 images from the Ultradeep layer of HSC-SSP PDR2.

The finetuned classifier is a simple linear model which takes 128x128 pixel images as input, passes them through the encoder, and outputs a number between 0 and 1. Outputs close to 1 indicate a galaxy which likely possesses tidal features.

## Code Usage
---
This code can be used to load the pre-trained model and reproduce the results of Desmons et al. (in prep) but is also designed to be easily customisable to use on your own data or to alter the augmentations used by the model. When using the pre-trained models or using your own data for the models, the only files that need to be looked at are the `training.py` and `model_testing.py` files.

---
### Using the Pre-trained Model

First, download the required data and pre-trained model from Zenodo which can be found at [https://zenodo.org/record/8146225](https://zenodo.org/record/8146225). This includes the `Datasets` folder containing the unlabelled dataset `Unlabelled_ds` and the tidal feature dataset `Tidal_ds`, the pre-trained encoder folder `Trained_NNCLR_model`, and the pre-trained finetuned classifier folder `Trained_finetuned_classifier`.

**model_testing.py** contains code to assemble these datasets and load the pre-trained model. It will apply the model to the test dataset and generate TPR, FPR and Threshold arrays as well as a ROC curve which will all be saved to a chosen location. The variables `dataset_path`, `NNCLR_load_path`, `Classifier_load_path`, and `save_path` will need to be edited to match your system setup and file structure.

---
### Training the Model on Different Data
When using the model on new data, the data should be formatted as a Tensorflow dataset. The only requirement for this dataset is that the image associated with a single dataset element should be able to be accessed using the format: `image = element["image"]`.

**training.py** contains the code used to assemble datasets and train and save the self-supervised NNCLR encoder and supervised finetuned classifier. This code gives the option to train both of these models or load a pre-trained encoder and train only the finetuned classifier. This option can be set using the `Train_NNCLR` parameter in the last function, which should be set to `True` when training both models or `False` when loading a pre-trained encoder model.

**model_testing.py** contains code to assemble the datasets and load the pre-trained model. It will apply the model to the test dataset and generate TPR, FPR and Threshold arrays as well as a ROC curve which will all be saved to a chosen location.

In both of these Python files, a number of variables will need to be changed to reflect your dataset stucture, system setup, and files used. These include: `dataset_path`, `splits`, `bands`, `data_sizes`, `input_shape`, `NNCLR_load_path`, `save_path`, and `Classifier_load_path`. See the `training.py` and `model_testing.py` files for details about these variables. Depending on your data, you may also want to edit other parameters relating to the image augmentation (e.g. `CROP_TO` and `jitter_max`) or model parameters (e.g. `temperature` and `queue_size`) - see the **HYPERPARAMETERS** section in the `training.py` and `model_testing.py` files for details about these variables.

---
### Further Editing of the Model
When using the pre-trained models or using your own data for the models, the only files that need to be looked at are the `training.py` and `model_testing.py` files, however you may wish to make other changes to the code depending on your aims. Below is a basic description of each Python file in this repository to facilitate this.
- `dataset_preparation.py`: Prepares two kinds of datasets, an unlabelled dataset for self-supervised training and a labelled datasets for classifier training. Splits this labelled dataset into training, validation, and testing sets.
- `augmentations.py`: Defines the data augmentations to be used by the model.
- `model_define.py`: Defines the NNCLR self-supervised model, adapted from https://keras.io/examples/vision/nnclr/
- `training.py`: Calls functions from the above files to prepare the datasets, define augmentations, and define NNCLR self-supervised model. Then trains and saves the self-supervised NNCLR model AND/OR a supervised finetuned classifier. Also saves a plot of the supervised classifier binary crossentropy loss and the training and validation loss arrays.
- `model_testing.py`: Calls functions from the above files to prepare the datasets, define augmentations, and define NNCLR self-supervised model. Then loads the pre-trained models and applies these to the testing dataset. Also saves a plot of the ROC curve and the threshold, TPR, and FPR arrays.

All files described above also contain docstrings and are commented to increase readability. If you have problems or questions regarding the code, please do not hesitate to contact me: a.desmons@unsw.edu.au 

## Abstract
---
Low surface brightness substructures around galaxies, known as tidal features, are a valuable tool in the detection of past or ongoing galaxy mergers, and their properties can answer questions about the progenitor galaxies involved in the interactions. The assembly of current tidal feature samples is primarily achieved using visual classification, making it difficult to assemble large samples of tidal features and draw accurate and statistically robust conclusions about the galaxy evolution process. With upcoming large optical imaging surveys such as the Vera C. Rubin Observatory’s Legacy Survey of Space and Time, predicted to observe billions of galaxies, it is imperative that we refine our methods of detecting and classifying samples of merging galaxies. This paper presents promising results from a self-supervised machine learning model, trained on data from the Ultradeep layer of the Hyper Suprime-Cam Subaru Strategic Program optical imaging survey, designed to automate the detection of tidal features. We find that self-supervised models are capable of detecting tidal features, and also that our model outperforms previous automated tidal feature detection methods. The previous state of the art method achieved 76\% completeness for 22\% contamination, while our model achieves considerably higher (96\%) completeness for the same level of contamination. We emphasise a number of advantages of self-supervised models over fully supervised models including maintaining excellent performance when using only 50 labelled examples for training, and the ability to perform similarity searches using a single example of a galaxy with tidal features.
