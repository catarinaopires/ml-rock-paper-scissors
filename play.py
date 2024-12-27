from random import choice

import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

from constants import COLORS, MOVES, REV_CLASS_MAP, WINS


def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def calculate_winner(move1, move2):
    """Assuming move1 is the User and move2 is the Computer"""
    if move1 == move2:
        return "Tie"

    if move1 == WINS[move2]:
        return "Computer"
    else:
        return "User"


def rescale_frame(frame, width=1280):
    percent = width / int(frame.shape[1])
    height = int(frame.shape[0] * percent)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


def get_user_move(frame, model):
    roi = frame[100:500, 100:500]
    img = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (227, 227))

    # Predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])

    return REV_CLASS_MAP[move_code]


def display_info(frame, user_move, computer_move, winner):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2

    cv.putText(
        frame,
        "Your Move: " + user_move,
        (50, 50),
        font,
        font_scale,
        COLORS["white"],
        thickness,
    )
    cv.putText(
        frame,
        "Computer's Move: " + computer_move,
        (750, 50),
        font,
        font_scale,
        COLORS["blue"],
        thickness,
    )
    cv.putText(
        frame, "Winner: " + winner, (400, 600), font, 2, COLORS["green"], thickness * 2
    )

    if computer_move != "none":
        icon = cv.imread(f"images/computer/{computer_move}.png")
        icon = cv.resize(icon, (400, 400))
        frame[100:500, 800:1200] = icon


def main():
    model = load_trained_model("rock-paper-scissors-model.h5")
    if model is None:
        return

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    prev_move = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = rescale_frame(frame)
        # Rectangle for user to play
        cv.rectangle(frame, (100, 100), (500, 500), COLORS["white"], 2)
        # Rectangle for computer to play
        cv.rectangle(frame, (800, 100), (1200, 500), COLORS["blue"], 2)

        user_move = get_user_move(frame, model)

        # Game (human vs computer)
        if prev_move != user_move:
            if user_move != "none":
                computer_move = choice(MOVES)
                winner = calculate_winner(user_move, computer_move)
            else:
                computer_move = "none"
                winner = "Waiting..."
        prev_move = user_move

        display_info(frame, user_move, computer_move, winner)
        cv.imshow("Rock Paper Scissors", frame)

        if cv.waitKey(10) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
