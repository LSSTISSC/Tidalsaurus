"""
Very similar structure to the training python file but instead of training models:
    - Loads the trained self-supervised encoder 
    - Loads the trained supervised finetuned classifier
    - Applies these to the testing set
    - Plots and saves ROC curve
    - Saves Threshold, TPR, and FPR arrays
"""

import sys
# If not running model_testing.py from folder where other python files are stored
# Add path to folder where other required python files are stored
#sys.path.append('/path/to/model/python/files')

from augmentations import augmenter
from model_define import NNCLR
from dataset_preparation import prepare_datasets

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
#import tensorflow_datasets as tfds

# HYPERPARAMETERS

# For the the Dataset
dataset_path = '/path/to/load/datasets' # Where saved datasets are
splits = ['Unlabelled_ds', 'Tidal_ds'] # Dataset splits
bands = ['g', 'r', 'i', 'z', 'y'] # List of channels of the image

# For augmentation
noise_range = [1,3] # Minimum and maximum values of uniform distribution for noise
jitter_max = 13 # Maximum pixels for jittering image centre
CROP_TO = 96 # Height/width of images after augmentation

# For Dataset Assembly
data_sizes = [300, 30, 50] # Classifier training, validation, testing set size by class 
                            # (i.e. 300 positive + 300 negative for training)
labelled_batch_size = 24 # Batch size when training classifier
unlabelled_batch_size = 220 # Batch size when training encoder

# Model parameters
input_shape = (128, 128, 5) # Shape of images given to the model
encoder_input_shape = (CROP_TO, CROP_TO, input_shape[2]) # Shape of images after augmentations
shuffle_buffer = 5000 # Shuffle buffer for shuffling datasets
temperature = 0.1 # Model temperature parameter
queue_size = 10000 # Number of examples in queue for self-supervised training
AUTOTUNE = tf.data.AUTOTUNE

# For loading pre-trained models and saving models
NNCLR_load_path = "/path/to/load/saved/NNCLR_encoder" # Where the pre-trained NNCLR model is saved
Classifier_load_path = "/path/to/load/saved/Finetuned_classifier" # Where the pre-trained Finetuned classifier is saved
save_path = "/path/to/save/outputs" # Where models, arrays, and plots will be saved
 
# Load and prepare datasets
unlabelled_train_dataset, labelled_train_dataset, \
    val_dataset, test_dataset, scale = \
    prepare_datasets(shuffle_buffer = shuffle_buffer, 
                     bands = bands, 
                     labelled_batch_size = labelled_batch_size, 
                     unlabelled_batch_size = unlabelled_batch_size, 
                     dataset_path = dataset_path, 
                     splits = splits, 
                     data_sizes = data_sizes
                     )

# Instantiate models
# Self-supervised NNCLR model
model = NNCLR(input_shape = input_shape,
              encoder_input_shape = encoder_input_shape, 
              temperature = temperature, 
              queue_size = queue_size,
              scale = scale,
              noise_range = noise_range,
              jitter_max = jitter_max,
              CROP_TO = CROP_TO
              )


# Supervised finetuned classifier
finetuning_model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        augmenter(input_shape, scale, noise_range, jitter_max, CROP_TO),
        model.encoder,
        layers.Dense(1,activation='sigmoid'),
    ],
    name="finetuning_model",
    )



def Test_model(model, finetuning_model, save_path,
               NNCLR_path, Classifier_path,
               datasets = [unlabelled_train_dataset, labelled_train_dataset, val_dataset, test_dataset]
               ):
    """
    Function that:
        - Loads the trained self-supervised encoder 
        - Loads the trained supervised finetuned classifier
        - Applies these to the testing set
        - Plots and saves ROC curve
        - Saves Threshold, TPR, and FPR arrays

    Parameters:
    -----------
    model: Instantiated self-supervised NNCLR model
    finetuning_model: Instantiated supervised finetuned model
    save_path: str
        Path to save models
    NNCLR_path: str
        Path to load saved NNCLR model
    Classifier_path: str
        Path to load saved supervised classifier
    datasets: List of Tensorflow datasets
        Default [unlabelled_train_dataset, labelled_train_dataset, val_dataset, test_dataset]
    """
    # Unpack datasets
    unlabelled_train_dataset, labelled_train_dataset, val_dataset, test_dataset = datasets

    # Load the saved self-supervised NNCLR model
    model.encoder = keras.models.load_model(NNCLR_path)
    # Compile model
    model.compile(contrastive_optimizer=keras.optimizers.Adam())

    # Load the saved supervised finetuned classifier
    finetuning_model = keras.models.load_model(Classifier_path)
    # Compile the supervised finetuned classifier
    finetuning_model.compile(optimizer=keras.optimizers.Adam(), 
                             loss=keras.losses.BinaryCrossentropy(), 
                             metrics=[keras.metrics.BinaryAccuracy(name="acc")])
    
    # Extract images and labels from testing dataset
    for images, labels in test_dataset:
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
    
    # Apply model to testing set
    predictions  = finetuning_model.predict(numpy_images).ravel()
    fpr, tpr, thresholds = roc_curve(numpy_labels, predictions)

    # Calculate area under the roc curve
    AUC = auc(fpr, tpr)

    #Save FPR and TPR Arrays
    FPR = save_path+"/FPR.npy"
    TPR = save_path+"/TPR.npy"
    Thresholds = save_path+"/Thresholds.npy"
    np.save(FPR,fpr)
    np.save(TPR,tpr)
    np.save(Thresholds,thresholds)

    # Plot and save ROC curve
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(AUC))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Finetuned Classifier ROC ')
    plt.legend(loc='lower right')
    ROC_name = save_path+"/ROC_curve.png"
    plt.savefig(ROC_name)

    return


Test_model(model = model, finetuning_model = finetuning_model, save_path = save_path,
           NNCLR_path = NNCLR_load_path, Classifier_path = Classifier_load_path,
           datasets = [unlabelled_train_dataset, labelled_train_dataset, val_dataset, test_dataset])  