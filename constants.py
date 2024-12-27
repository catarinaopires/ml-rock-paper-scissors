# Define color constants for use in the application
COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
}

# Define possible moves in the game
MOVES = ["rock", "paper", "scissors"]
# Define class labels for classification
LABELS = MOVES + ["none"]

# Create a mapping from class labels to numerical indices
CLASS_MAP = {label: idx for idx, label in enumerate(LABELS)}
# Create a reverse mapping from numerical indices to class labels
REV_CLASS_MAP = {idx: label for label, idx in CLASS_MAP.items()}

# Define the winning relationships between moves
WINS = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}
