import os
import shutil
import random

SOURCE_DIR = "dataset/tomato/grading"

OUTPUT_DIR = "dataset/tomato/grading_split"

GRADES = ["A", "B", "C"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

for grade in GRADES:

    source_folder = os.path.join(SOURCE_DIR, grade)

    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split, split_images in [
        ("train", train_images),
        ("validation", val_images),
        ("test", test_images)
    ]:

        destination = os.path.join(
            OUTPUT_DIR,
            split,
            grade
        )

        os.makedirs(destination, exist_ok=True)

        for image in split_images:

            source_path = os.path.join(
                source_folder,
                image
            )

            destination_path = os.path.join(
                destination,
                image
            )

            shutil.copy2(
                source_path,
                destination_path
            )

    print(
        f"Grade {grade}: "
        f"Train={len(train_images)}, "
        f"Validation={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset splitting completed!")