"""
Defines the NNCLR model, adapted from https://keras.io/examples/vision/nnclr/
"""

from augmentations import augmenter

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
import resnet_cifar10_v2

N = 2
DEPTH = N*9+2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1

PROJECT_DIM = 128
WEIGHT_DECAY = 0.0005

def encoder(encoder_input_shape):
    """
    Encoder to encode images into lower-dimensional representations

    Parameters:
    ---------
    encoder_input_shape: tuple of ints of length 3
            Shape of the images after augmentations
            Fomat: (width, height, channels)
            Usually: (CROP_TO, CROP_TO, channels)
    """
    inputs = layers.Input(encoder_input_shape, name="encoder_input")
    x = resnet_cifar10_v2.stem(inputs)
    x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
    x = layers.GlobalAveragePooling2D(name="backbone_pool")(x)

    # Projection head.
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(
        PROJECT_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    outputs = layers.BatchNormalization()(x)
    return tf.keras.Model(inputs, outputs, name="encoder")

class NNCLR(keras.Model):
    def __init__(self, input_shape, encoder_input_shape, temperature, queue_size, 
                 scale, noise_range, jitter_max, CROP_TO):
        """
        Model which augments and encodes images and calculates the loss

        Parameters:
        -----------
        input_shape: tuple of ints of length 3
            Shape of the images
            Fomat: (width, height, channels)
        encoder_input_shape: tuple of ints of length 3
            Shape of the images after augmentations
            Fomat: (width, height, channels)
            Usually: (CROP_TO, CROP_TO, channels)
        scale: List of floats of length number of channels/bands
            Scaling factor by band/channel
        noise_range: List of ints of length 2
            Specifies minimum and maximum values of uniform distribution for noise
            Format: [minval, maxval]
        jitter_max: int
            Maximum pixels for jittering image centre
        CROP_TO: int
            Final size of image
        """
        super().__init__()
        # Define metrics
        self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Define augmenter
        self.contrastive_augmenter = augmenter(input_shape, scale, noise_range, jitter_max, CROP_TO)

        # Define model structure
        self.encoder = encoder(encoder_input_shape)
        self.projection_head = keras.Sequential(
            [
                layers.Input(shape=(PROJECT_DIM,), name="projection_input"), 
                layers.Dense(PROJECT_DIM, activation='relu'),
                layers.Dense(PROJECT_DIM),
            ],
            name="projection_head"
        )
        self.temperature = temperature

        # Initialise the support set that will be used to find nearest 
        # neighbours
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, PROJECT_DIM)), axis=1
            ),
            trainable=False
        )

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
    
    def nearest_neighbour(self, projections):
        """
        Find the nearest neighbour to `projections` in the support set 
        `self.feature_queue`
        """
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )
        # Get the feature vector (projection) in the feature queue with the 
        # highest support_similarity
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )

        # Need to do tf.stop_gradient to treat this as a constant and stop
        # backprop from potentially flowing back to the wrong input?
        return projections + tf.stop_gradient(nn_projections - projections)
    
    def update_contrastive_accuracy(self, features_1, features_2):
        # Calculate the similarity between features 1 and 2
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        # For each batch element, we want most similar vector in other 
        # set to be the corresponding nearest neighbour from support set
        # (i.e. they have the same index, should be along diagonal in 
        # `similarities`).
        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0)
        )

    def update_correlation_accuracy(self, features_1, features_2):
        # Correlation accuracy measures how redundant the features are by
        # measuring how similar the activations on different images are.
        # Ideally, we would want no dependence between different features

        # Standardise feature vectors
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        # Get cross-correlation matrix (like similarities of feature activations 
        # over different images?)
        batch_size = tf.cast(tf.shape(features_1)[0], tf.float32)
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        # Update the correlation accuracy
        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0)
        )

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        # Calculate cosine similarities (scaled by temperature) of:
        # - nearest neighbour of projection_1 to projection_2
        # - projection_2 to nearest neighbour of projection_1
        # - nearest neighbour of projection_2 to projection_1
        # - projection_1 to nearest neighbour of projection_2
        similarities_1_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )

        similarities_1_2_2 = (
            tf.matmul(
                projections_2, self.nearest_neighbour(projections_1), transpose_b=True
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )

        similarities_2_1_2 = (
            tf.matmul(
                projections_1, self.nearest_neighbour(projections_2), transpose_b=True
            )
            / self.temperature
        )

        # Calculate the contrastive loss
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat(
                [
                    contrastive_labels, 
                    contrastive_labels, 
                    contrastive_labels, 
                    contrastive_labels
                ],
                axis=0
            ),
            tf.concat(
                [
                    similarities_1_2_1,
                    similarities_1_2_2,
                    similarities_2_1_1,
                    similarities_2_1_2,
                ],
                axis=0
            ),
            from_logits=True
        )

        # Update the feature queue
        self.feature_queue.assign(
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )

        return loss

    def train_step(self, images):
        # Get the augmented images
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)

        # Pass through the model and calculate the loss
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        
        # Calculate gradients and backpropagate
        gradients = tape.gradient(
            contrastive_loss, 
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights
            )
        )
        
        # Update accuracy metrics
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)
        
        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result()
        }
