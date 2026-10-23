import cv2
import sys
import os


def validate_image(image_path):

    print("\n========================================")
    print("        IMAGE VALIDATION")
    print("========================================")

    # --------------------------------------------------
    # 1. Check whether file exists
    # --------------------------------------------------

    if not os.path.exists(image_path):
        print("❌ Image file not found.")
        return False

    # --------------------------------------------------
    # 2. Read image
    # --------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Image could not be opened.")
        return False

    print("✅ Image successfully loaded.")

    # --------------------------------------------------
    # 3. Check image dimensions
    # --------------------------------------------------

    height, width = image.shape[:2]

    print(f"Image size: {width} x {height}")

    if width < 224 or height < 224:
        print("❌ Image resolution is too low.")
        return False

    print("✅ Image resolution is acceptable.")

    # --------------------------------------------------
    # 4. Convert to grayscale
    # --------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------
    # 5. Check brightness
    # --------------------------------------------------

    brightness = gray.mean()

    print(f"Brightness: {brightness:.2f}")

    if brightness < 40:
        print("❌ Image is too dark.")
        return False

    if brightness > 220:
        print("❌ Image is too bright.")
        return False

    print("✅ Lighting is acceptable.")

    # --------------------------------------------------
    # 6. Check blur / sharpness
    # --------------------------------------------------

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    print(f"Sharpness score: {blur_score:.2f}")

    if blur_score < 14.56:
        print("❌ Image appears too blurry.")
        return False

    print("✅ Image sharpness is acceptable.")

    # --------------------------------------------------
    # 7. Final result
    # --------------------------------------------------

    print("----------------------------------------")
    print("✅ IMAGE VALIDATION PASSED")
    print("The image is suitable for further processing.")
    print("========================================")

    return True


# ------------------------------------------------------
# Main program
# ------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Please provide an image path.")

        print(
            'Example: python src/image_validation.py '
            '"C:\\path\\to\\tomato.jpg"'
        )

        sys.exit()

    image_path = sys.argv[1]

    validate_image(image_path)