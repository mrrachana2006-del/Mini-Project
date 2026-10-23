from pathlib import Path
import random
import shutil

# Original dataset location
SOURCE_DIR = (
    Path.home()
    / "Downloads"
    / "Tomato Maturity Detection and Quality Grading Dataset"
    / "Tomato Maturity Detection and Quality Grading Dataset"
    / "Tomato Quality Grading Dataset"
    / "Tomato Quality Grading Dataset"
    / "Original Dataset"
)
# Our project dataset folder
DEST_DIR = Path("dataset") / "tomato"

# Split ratio
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# Dataset class names
classes = {
    "Fresh": "fresh",
    "Rotten": "rotten"
}
# Check source folder
if not SOURCE_DIR.exists():
    print("Source folder not found!")
    print("Check the SOURCE_DIR path in the code.")
    exit()

for source_class, output_class in classes.items():

    source_folder = SOURCE_DIR / source_class

    if not source_folder.exists():
        print(f"Folder not found: {source_folder}")
        continue

    # Get image files
    images = [
        file for file in source_folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    # Shuffle images
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    splits = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        destination = DEST_DIR / split_name / output_class
        destination.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, destination / image.name)

    print(f"\n{source_class}:")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(validation_images)}")
    print(f"Test: {len(test_images)}")

print("\nDataset splitting completed successfully!")