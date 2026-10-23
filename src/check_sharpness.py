import cv2
import os
import numpy as np

folder = "dataset/tomato/grading_split/train"

scores = []

for grade in ["A", "B", "C"]:

    grade_folder = os.path.join(folder, grade)

    for filename in os.listdir(grade_folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(grade_folder, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        scores.append(score)


threshold = np.percentile(scores, 10)

print("========================================")
print("       DATASET SHARPNESS ANALYSIS")
print("========================================")

print("Images analyzed:", len(scores))
print(f"Minimum sharpness: {min(scores):.2f}")
print(f"Maximum sharpness: {max(scores):.2f}")
print(f"Average sharpness: {np.mean(scores):.2f}")
print(f"Median sharpness: {np.median(scores):.2f}")
print(f"10th percentile: {threshold:.2f}")
print("========================================")
print("Recommended initial threshold:", round(threshold, 2))
print("========================================")