# Rock Paper Scissors Game

This project implements a Rock Paper Scissors game using computer vision and machine learning. The project consists of three main steps:
1. Collecting the dataset (gather images for each gesture: rock, paper, scissors, and none).
2. Training the model.
3. Playing the game with the computer.

## Prerequisites

- Python 3.x
- OpenCV
- TensorFlow
- Keras
- NumPy
- scikit-learn

## Step 1: Collecting the Dataset
To collect the dataset, you need to gather images for each gesture (rock, paper, scissors, and none). Use the `collect_dataset.py` script to capture images from your webcam.

### Usage
```python collect_dataset.py <label_name> <num_samples>```
- <label_name>: The name of the gesture (e.g., rock, paper, scissors, none).
- <num_samples>: The number of images to capture.
### Example
To collect 100 images for the "rock" gesture: `python collect_dataset.py rock 100`

## Step 2: Training the Model
Once you have collected the dataset, you can train the model using the `train.py` script.

### Usage
```python train.py```

This script will load the images, preprocess the data, and train a convolutional neural network (CNN) model. The trained model will be saved as `rock-paper-scissors-model.h5`.

## Step 3: Playing the Game
After training the model, you can play the game with the computer using the `play.py` script.

### Usage
```python play.py```

This script will load the trained model and start the webcam. You can play the game by showing your gesture (rock, paper, or scissors) to the webcam. The computer will randomly choose a gesture, and the winner will be displayed on the screen.

## Directory Structure
```
rock-paper-scissors/
├── collect_dataset.py
├── constants.py
├── play.py
├── train.py
├── images/
│   ├── rock/
│   ├── paper/
│   ├── scissors/
│   └── none/
└── rock-paper-scissors-model.h5
```
- `collect_dataset.py`: Script to collect images for the dataset.
- `constants.py`: Contains constants used in the application.
- `play.py`: Script to play the game with the computer.
- `train.py`: Script to train the model.
- `images`: Directory to store the collected images.
- `rock-paper-scissors-model.h5`: Trained model file.