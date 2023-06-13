"""
Train and save a self-supervised NNCLR model AND/OR a supervised finetuned classifier
The final function:
    - Trains one or both models
    - Saves the models to the designated save path
    - Saves a plot of the supervised classifier binary crossentropy loss
    - Saves the training and validation loss arrays
"""

import sys
# If not running training.py from folder where other python files are stored
# Add path to folder where other required python files are stored
#sys.path.append('/path/to/model/python/files')

from augmentations import augmenter
from model_define import NNCLR
from dataset_preparation import prepare_datasets

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import numpy as np
#import tensorflow_datasets as tfds

# HYPERPARAMETERS

# For the the Dataset
dataset_path = '/path/to/load/datasets' # Where saved datasets are
splits = ['Unlabelled_ds', 'Tidal_ds'] # Dataset splits
bands = ['g', 'r', 'i', 'z', 'y'] # List of channels (in order) of the image

# For augmentation
noise_range = [1,3] # Minimum and maximum values of uniform distribution for noise
jitter_max = 13 # Maximum pixels for jittering image centre
CROP_TO = 96 # Height/width of images after augmentation

# For training
num_epochs = 25 # Number of epochs for training self-supervised NNCLR model
num_epochs_sup = 50 # Number of epochs for training supervised finetuned classifier
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



def train_or_load(Train_NNCLR, model, finetuning_model, num_epochs, 
                  num_epochs_sup, save_path, NNCLR_path = None,
                  datasets = [unlabelled_train_dataset, labelled_train_dataset, val_dataset, test_dataset]
                  ):
    """
    Function that offers two options:
        1. Train self-supervised NNCLR model and supervised finetuned classifier
        2. Load self-supervised NNCLR model and train supervised finetuned classifier
    And plots the loss + saves the trained models

    Parameters:
    -----------
    Train_NNCLR: True or False
        If True: Trains self-supervised NNCLR model and supervised finetuned classifier
        If False: Loads self-supervised NNCLR model and trains supervised finetuned classifier
    model: Instantiated self-supervised NNCLR model
    finetuning_model: Instantiated supervised finetuned model
    num_epochs: int
        number of epochs to train self-supervised NNCLR model
    num_epochs_sup: int
        number of epochs to train supervised finetuned classifier
    save_path: str
        Path to save models
    NNCLR_path: str
        Path to load saved model
        Default = None
    datasets: List of Tensorflow datasets
        Default [unlabelled_train_dataset, labelled_train_dataset, val_dataset, test_dataset]
    """
    # Unpack datasets
    unlabelled_train_dataset, labelled_train_dataset, val_dataset, test_dataset = datasets

    # If Train_NNCLR == True then train the self-supervised NNCLR model
    if Train_NNCLR == True:
        # Compile the self-supervised NNCLR model
        model.compile(contrastive_optimizer=keras.optimizers.Adam())
        # Train
        model_history = model.fit(unlabelled_train_dataset, epochs=num_epochs)
        # Save model as "NNCLR_encoder"
        model_name = save_path+"/NNCLR_encoder"
        model_to_save = model.encoder
        model_to_save.save(model_name)

    # Otherwise load the self-supervised NNCLR model
    else:
        # Load saved self-supervised NNCLR model
        model.encoder = keras.models.load_model(NNCLR_path)
        # Compile model
        model.compile(contrastive_optimizer=keras.optimizers.Adam())
    
    # Compile the supervised finetuned classifier
    finetuning_model.compile(optimizer=keras.optimizers.Adam(), 
                             loss=keras.losses.BinaryCrossentropy(), 
                             metrics=[keras.metrics.BinaryAccuracy(name="acc")])
    # Train the supervised finetuned classifier
    finetuned_history = finetuning_model.fit(labelled_train_dataset, 
                                             epochs = num_epochs_sup, 
                                             validation_data = val_dataset)
    
    # Plot the loss and save plot
    plt.figure()
    plt.plot(finetuned_history.history["loss"], label = 'train')
    plt.plot(finetuned_history.history["val_loss"],label = 'validation')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Crossentropy')
    plt.title("Classifier Loss")
    plt.legend(loc='upper right')
    Loss_name = save_path + "/Classifier_Loss.png"
    plt.savefig(Loss_name)

    # Save the training and validation loss arrays
    Train_loss = np.asarray(finetuned_history.history["loss"])
    Train_loss_name = save_path+"/Train_loss.npy"
    np.save(Train_loss_name,Train_loss)

    Val_loss = np.asarray(finetuned_history.history["val_loss"])
    Val_loss_name = save_path+"/Val_loss.npy"
    np.save(Val_loss_name,Val_loss)

    # Save finetuned classifier model
    Model_name = save_path+"/Finetuned_classifier"
    finetuning_model.save(Model_name)

    return

train_or_load(Train_NNCLR = True, model = model, finetuning_model = finetuning_model, 
              num_epochs = num_epochs, num_epochs_sup = num_epochs_sup, save_path = save_path,
              NNCLR_load_path = NNCLR_load_path,
              datasets = [unlabelled_train_dataset, labelled_train_dataset, val_dataset, test_dataset])  