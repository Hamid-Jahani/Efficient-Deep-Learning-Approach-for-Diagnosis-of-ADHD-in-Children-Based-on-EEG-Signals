"""CNN and customised-ResNet model definitions, extracted from ADHD.ipynb."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform, random_uniform
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model


def cnn_model(input_shape=(64, 64, 3), classes: int = 6) -> Model:
    """A CNN with dropout regularisation (``CNN_model_1`` in the notebook)."""
    initializer = tf.keras.initializers.HeNormal()
    X_input = Input(input_shape)

    X = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer)(X_input)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(rate=0.25)(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), padding="valid", kernel_initializer=initializer)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(rate=0.25)(X)

    X = Conv2D(128, (2, 2), strides=(1, 1), padding="valid", kernel_initializer=initializer)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(rate=0.25)(X)

    X = Flatten()(X)
    X = Dense(128, activation="relu", kernel_initializer=initializer)(X)
    X = Dropout(rate=0.25)(X)
    X = Dense(classes, activation="sigmoid", kernel_initializer=initializer)(X)

    return Model(inputs=X_input, outputs=X, name="cnn_model")


def identity_block(X, f: int, filters, training: bool = True, initializer=random_uniform):
    """Identity residual block (no dimension change on the shortcut path)."""
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=1, strides=(1, 1), padding="valid", kernel_initializer=initializer(seed=0))(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding="same", kernel_initializer=initializer(seed=0))(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding="valid", kernel_initializer=initializer(seed=0))(X)

    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)
    return X


def convolutional_block(X, f: int, filters, s: int = 2, training: bool = True, initializer=glorot_uniform):
    """Residual block with a strided/projected shortcut path."""
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding="valid", kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation("relu")(X)

    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding="same", kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation("relu")(X)

    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding="valid", kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)

    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding="valid", kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def resnet_model(input_shape=(64, 64, 3), classes: int = 6) -> Model:
    """The customised ResNet (``CNN_model_2`` in the notebook): plain conv
    stages interleaved with identity residual blocks."""
    initializer = tf.keras.initializers.HeNormal()
    X_input = Input(input_shape)

    X = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer)(X_input)
    X = identity_block(X, 3, [32, 32, 32])
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(rate=0.25)(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), padding="valid", kernel_initializer=initializer)(X)
    X = identity_block(X, 3, [64, 64, 64])
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(rate=0.25)(X)

    X = Conv2D(128, (2, 2), strides=(1, 1), padding="valid", kernel_initializer=initializer)(X)
    X = identity_block(X, 3, [128, 128, 128])
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(rate=0.25)(X)

    X = Flatten()(X)
    X = Dense(128, activation="relu", kernel_initializer=initializer)(X)
    X = Dropout(rate=0.25)(X)
    X = Dense(classes, activation="sigmoid", kernel_initializer=initializer)(X)

    return Model(inputs=X_input, outputs=X, name="resnet_model")
