import os
import sys

import cv2 as cv

from constants import COLORS


def parse_arguments():
    try:
        label_name = sys.argv[1]
        num_samples = int(sys.argv[2])
        return label_name, num_samples
    except:
        print("Arguments missing.")
        print("Usage: python collect_dataset.py <label_name> <num_samples>")
        exit(-1)


def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory {path} created.")
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        exit(-1)


def main():
    label_name, num_samples = parse_arguments()
    image_collection_path = os.path.join("images", label_name)
    create_directory(image_collection_path)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    start = False
    image_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            continue

        if image_counter == num_samples:
            break

        cv.rectangle(frame, (100, 100), (300, 300), COLORS["white"], 2)
        if start:
            roi = frame[100:300, 100:300]
            image_name = os.path.join(
                image_collection_path, f"image_{image_counter}.png"
            )
            cv.imwrite(image_name, roi)
            image_counter += 1

        font_scale = 0.7
        thickness = 2
        cv.putText(
            frame,
            f"Collecting {image_counter}/{num_samples}",
            (5, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLORS["yellow"],
            thickness,
        )
        cv.putText(
            frame,
            "'s' to start/stop, 'q' to quit",
            (5, 80),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLORS["yellow"],
            thickness,
        )
        cv.imshow("Collecting images...", frame)

        key = cv.waitKey(1)
        if key == ord("s"):
            start = not start
        elif key == ord("q"):
            break

    print(f"\n{image_counter} image(s) saved to {image_collection_path}")
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
