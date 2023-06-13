"""
Define data augmentations to be used by the model. These include:
    - Randomly adding gaussion noise to the image
    - Randomly cropping the image, jittering the centre of the image
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

class RandomGaussianNoise(layers.Layer):
    """
    Randomly add Gaussian noise
    """
    def __init__(self, scale, noise_range):
        """
        Parameters:
        -----------
        scale: List of floats of length number of channels/bands
            Scaling factor by band/channel
        noise_range: List of ints of length 2
            Specifies minimum and maximum values of uniform distribution for noise
            Format: [minval, maxval]
        
        Returns:
        -------
        images: numpy image
            Images with noise added
        """
        super().__init__()
        self.scale = scale
        self.noise_range = noise_range

    def call(self, images):
        # Add some random noise
        sigma = tf.random.uniform([], minval = self.noise_range[0], maxval = self.noise_range[1])
        images += sigma * self.scale * tf.random.normal(tf.shape(images))
        return images

class RandomResizedCrop(layers.Layer):
    """
    Randomly crop the images

    Parameters:
        -----------
        scale: List of floats of length number of channels/bands
            Scaling factor by band/channel
        jitter_max: int
            Maximum pixels for jittering image centre
        CROP_TO: int
            Final size of image
        
        Returns:
        -------
        images: numpy image
            Randomly cropped images
    """

    def __init__(self, scale, jitter_max,CROP_TO):
        """
        Parameters:
        -----------
        scale: List of floats of length number of channels/bands
            Scaling factor by band/channel
        jitter_max: int
            Maximum pixels for jittering image centre
        CROP_TO: int
            Final size of image
        
        Returns:
        -------
        images: numpy image
            Randomly cropped images
        """
        super().__init__()
        self.scale = scale
        self.PRE_CROP = CROP_TO+jitter_max
        self.CROP_TO = CROP_TO

    def call(self, images):
        batch_size = tf.shape(images)[0]
        im_height = tf.shape(images)[1]
        im_channels = tf.shape(images)[3]
        
        # Crop and resize
        images = tf.image.central_crop(images,self.PRE_CROP/im_height)
        images = tf.image.random_crop(images, (batch_size,self.CROP_TO, self.CROP_TO, im_channels))

        return images

def augmenter(input_shape, scale, noise_range, jitter_max, CROP_TO):
    """
    Apply augmentations

    Parameters:
    -----------
    input_shape: tuple of ints of length 3
        Shape of the images
        Fomat: (width, height, channels)
    scale: List of floats of length number of channels/bands
        Scaling factor by band/channel
    noise_range: List of ints of length 2
        Specifies minimum and maximum values of uniform distribution for noise
        Format: [minval, maxval]
    jitter_max: int
        Maximum pixels for jittering image centre
    CROP_TO: int
        Final size of image
    
    Returns:
    -------
    images: numpy image
        Fully augmented images
    """
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.RandomFlip(mode="horizontal_and_vertical"),
            RandomGaussianNoise(scale = scale, noise_range=noise_range),
            RandomResizedCrop(scale=scale, jitter_max=jitter_max, CROP_TO = CROP_TO),
        ]
    )