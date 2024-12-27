import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import (Activation, Convolution2D, Dropout,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from constants import CLASS_MAP


def get_model():
    # Transfer learning (use a pre-trained model as a base and fine-tune it)
    # This model architecture leverages the power of
    # a pre-trained CNN for feature extraction (without classidication layers)
    # adds a layer to adapt those features to our specific classification task,
    # and uses global average pooling for efficiency.
    # Dropout helps prevent overfitting, and the final softmax layer provides class probabilities for prediction
    model = Sequential(
        [
            ResNet50(input_shape=(227, 227, 3), include_top=False),
            Dropout(0.5),
            Convolution2D(
                len(CLASS_MAP), (1, 1), padding="valid"
            ),  # "adapter" layer to match the number of classes
            Activation("relu"),
            GlobalAveragePooling2D(),  # classification block
            Activation("softmax"),  # classification block
        ]
    )
    return model


def load_images(image_collection_path):
    dataset = []
    for directory in os.listdir(image_collection_path):
        path = os.path.join(image_collection_path, directory)
        if not os.path.isdir(path) or directory not in CLASS_MAP:
            continue
        for item in os.listdir(path):
            # Ignore unwanted files
            if item.startswith("."):
                continue
            img_path = os.path.join(path, item)
            img = cv.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (227, 227))
            dataset.append([img, directory])
    return dataset


def preprocess_data(dataset):
    """
    dataset = [
        [[...], 'rock'],
        [[...], 'paper'],
        ...
    ]
    """
    data, labels = zip(*dataset)
    labels = [CLASS_MAP[label] for label in labels]
    labels = to_categorical(labels)
    return np.array(data), np.array(labels)


def main():
    print("Loading images...")
    dataset = load_images("images/")
    if not dataset:
        print("Error: No images loaded.")
        return

    data, labels = preprocess_data(dataset)

    model = get_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.1, random_state=1
    )

    print("Training model...")
    model.fit(
        train_data, train_labels, epochs=10, validation_data=(val_data, val_labels)
    )

    model.save("rock-paper-scissors-model.h5")
    print("Model saved as rock-paper-scissors-model.h5")


if __name__ == "__main__":
    main()
