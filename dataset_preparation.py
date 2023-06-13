"""
Prepares two kinds of datasets:
    - An unlabelled dataset for self-supervised training
    - Labelled datasets for classifier training 

 The functions in this file are:
    - load_datasets: Load two dataset, one large unlabelled one and one with known tidal features
    - find_mad_std: Find the median absolute deviation of each channel to use for normalisationg and augmentations
    - normalisation: Normalises images
    - split_datasets: Split datasets into training, validation, and testing for classifier training
    - label_positive and label_negative: Assigns labels to images
    - prepare_datasets: Uses the above functions to assemble the datasets
"""

import os
import astropy
from astropy.stats import mad_std

import tensorflow as tf
import numpy as np


def load_datasets(dataset_path, splits):
    """
    Load two dataset, one large unlabelled one and one with known tidal features

    Parameters
    ----------
    dataset_path: str
        Path location of datasets
    splits: List of str
        Name of datasets to load from dataset_path
        Format: ['Unlabelled_ds', 'Tidal_ds']

    Returns
    -------
    Unlabelled_dset: Tensorflow Dataset
        Large unlabelled dataset
    Tidal_dset: Tensorflow Dataset
        Dataset with known tidal features
    """
    # Load unlabelled large (40,000 gals) dataset
    path =os.path.join(dataset_path, splits[0]) 
    Unlabelled_dset = tf.data.Dataset.load(path,compression=None)

     # Load small tidal feature (380 gals) dataset
    path =os.path.join(dataset_path, splits[1]) 
    Tidal_dset = tf.data.Dataset.load(path,compression=None)

    return Unlabelled_dset, Tidal_dset

def find_mad_std(dataset,bands):
    """
    Find the median absolute deviation of each channel to use for normalisationg and augmentations

    Parameters
    ----------
    dataset: Tensorflow Dataset
        Dataset to be sampled from
    bands: list of str
        List of channels of the image e.g. ['g', 'r', 'i']

    Returns
    -------
    scaling: List of floats of length len(bands)
        Scaling factor by band/channel
    """
    cutouts = []
    # Append the first 1000 galaxies from the dataset to an array
    for (batch, entry) in enumerate(dataset.take(1000)):
        # Only select the image 
        cutouts.append(entry['image'])
    cutouts = np.stack(cutouts)

    # add median absolute deviation for each band to an array
    scaling = []
    for i, b in enumerate(bands):
        sigma = mad_std(cutouts[..., i].flatten())
        scaling.append(sigma)
    
    return scaling

def normalisation(example,scale):
    """
    Normalises images
    example: element in dataset
    scale: array of median absolute deviation for each band

    Parameters
    ----------
    example: Dataset item
        Element in dataset
    scale: List of floats of length len(bands)
        Scaling factor by band/channel

    Returns
    -------
    img: numpy array
        normalised image
    """
    # Get only the image from the dataset element
    img = example['image']
    img = tf.math.asinh(img / tf.constant(scale, dtype=tf.float32) / 3.)
    # We return the normalised images
    return img

def split_datasets(Unlabelled_dset, Tidal_dset, data_sizes):
    """
    Split datasets into training, validation, and testing for classifier training

    Parameters
    ----------
    Unlabelled_dset: Tensorflow Dataset
        Large unlabelled dataset
    Tidal_dset: Tensorflow Dataset
        Dataset with known tidal features
    data_sizes: List of ints of length 3
        Dataset sizes per class to use for training, validation, and testing
        Format: [train_size, val_size, test_size]
        For class = 2 (tidal, no_tidal), train_size should be half the total
        training set size. 
    
    Returns
    -------
    No_tidal_dsets: List of Tensorflow Datasets
        Negative examples to use for training, validation, and testing
        Format: [train_dset, val_dset, test_dset]
    Tidal_dsets: List of Tensorflow Datasets
        Positive examples to use for training, validation, and testing
        Format: [train_dset, val_dset, test_dset]
    """
    
    # Unpack data_sizes
    train_size, val_size, test_size = data_sizes    

    # Split the dataset without tidal features
    no_tidal_train_dset = Unlabelled_dset.take(train_size)
    no_tidal_val_dset = Unlabelled_dset.skip(train_size).take(val_size)
    no_tidal_test_dset = Unlabelled_dset.skip(train_size + val_size).take(test_size)

    # Split the dataset with tidal features
    tidal_train_dset = Tidal_dset.take(train_size)
    tidal_val_dset = Tidal_dset.skip(train_size).take(val_size)
    tidal_test_dset = Tidal_dset.skip(train_size + val_size).take(test_size)

    No_tidal_dsets = [no_tidal_train_dset,no_tidal_val_dset,no_tidal_test_dset]
    Tidal_dsets = [tidal_train_dset,tidal_val_dset,tidal_test_dset]
    return [No_tidal_dsets, Tidal_dsets]


# Functions to assign positive (tidal) and negative (non-tidal) labels
def label_positive(image):
    return image, 1

def label_negative(image):
    return image, 0


def prepare_datasets(shuffle_buffer, bands, labelled_batch_size, 
                     unlabelled_batch_size, dataset_path, splits, data_sizes
                     ):
    """
    Prepares datasets for training, validation, and testing using the functions above

    Parameters
    ----------
    shuffle_buffer: int
        Buffer size to use when shuffling datasets
    bands: list of str
        List of channels of the image e.g. ['g', 'r', 'i']
    labelled_batch_size: int
        Batch size to use when training supervised classifier 
    unlabelled_batch_size: int
        Batch size to use when training self-supervised encoder 
    dataset_path: str
        Path location of datasets
    splits: List of str
        Name of datasets to load from dataset_path
        Format: ['No_tidal', 'Tidal_ds_no_dup']
    data_sizes: List of ints of length 3
        Dataset sizes per class to use for training, validation, and testing
        Format: [train_size, val_size, test_size]
        For class = 2 (tidal, no_tidal), train_size should be half the total
        training set size. 
    
    Returns
    -------
    unlabelled_train_dataset: Batched Tensorflow Dataset
        Dataset to use for training the self-supervised encoder
    labelled_train_dataset: Batched Tensorflow Dataset
        Dataset to use for training the supervised classifier
    val_dataset: Batched Tensorflow Dataset
        Dataset to use for validation of the supervised classifier
    test_dataset: Batched Tensorflow Dataset
        Dataset to use for testing of the supervised classifier
    scale: List of floats of length len(bands)
        Scaling factor by band/channel
    """
    
    train_size, val_size, test_size = data_sizes

    # Load datasets and find scaling factor
    Unlabelled_dset, Tidal_dset = load_datasets(dataset_path, splits)
    scale = find_mad_std(Unlabelled_dset,bands = bands)

    # Normalise datasets
    norm_unlabelled_dset = Unlabelled_dset.map(lambda x: normalisation(x, scale))
    norm_Tidal_dset = Tidal_dset.map(lambda x: normalisation(x, scale))

    #Split datasets
    No_tidal_dsets, Tidal_dsets = split_datasets(norm_unlabelled_dset, norm_Tidal_dset, data_sizes)
    
    no_tidal_train_dset, no_tidal_val_dset, no_tidal_test_dset = No_tidal_dsets
    tidal_train_dset, tidal_val_dset, tidal_test_dset = Tidal_dsets

    # Assign labels to datasets
    positive_dset_train = tidal_train_dset.map(label_positive)
    negative_dset_train = no_tidal_train_dset.map(label_negative)

    positive_dset_val = tidal_val_dset.map(label_positive)
    negative_dset_val = no_tidal_val_dset.map(label_negative)

    positive_dset_test = tidal_test_dset.map(label_positive)
    negative_dset_test = no_tidal_test_dset.map(label_negative)

    # Combine positive and negative datasets
    labelled_dset_train = positive_dset_train.concatenate(negative_dset_train)
    labelled_dset_val = positive_dset_val.concatenate(negative_dset_val)
    labelled_dset_test = positive_dset_test.concatenate(negative_dset_test)

    # Batch and shuffle datasets
    labelled_train_dataset = (labelled_dset_train
                             .shuffle(buffer_size=shuffle_buffer)
                             .batch(labelled_batch_size, drop_remainder=True)
                             )

    val_dataset = (labelled_dset_val
                   .batch(val_size*2)
                   .shuffle(buffer_size=shuffle_buffer)
                   )
    
    test_dataset = (labelled_dset_test
                   .batch(test_size*2)
                   )
    
    unlabelled_train_dataset = (norm_Tidal_dset
                                .concatenate(norm_unlabelled_dset)
                                .shuffle(buffer_size=shuffle_buffer)
                                .batch(unlabelled_batch_size, drop_remainder=True)
                                )

    return unlabelled_train_dataset, labelled_train_dataset, val_dataset, test_dataset, scale