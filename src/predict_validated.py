import sys
import torch
from torchvision import models, transforms
from PIL import Image

from image_validation import validate_image


# --------------------------------------------------
# 1. Check command-line argument
# --------------------------------------------------

if len(sys.argv) != 2:

    print("Please provide an image path.")

    print(
        'Example: python src/predict_validated.py '
        '"C:\\path\\to\\tomato.jpg"'
    )

    sys.exit()


image_path = sys.argv[1]


# --------------------------------------------------
# 2. Image Validation
# --------------------------------------------------

is_valid = validate_image(image_path)

if not is_valid:

    print("\n❌ Processing stopped.")
    print("Please upload or capture a clearer image.")

    sys.exit()


# --------------------------------------------------
# 3. Load trained grading model
# --------------------------------------------------

print("\nSTEP 2: AI QUALITY GRADING")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

classes = ["A", "B", "C"]


model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    3
)


model.load_state_dict(
    torch.load(
        "tomato_grading_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# 4. Image Transformation
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


image = Image.open(image_path).convert("RGB")

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# --------------------------------------------------
# 5. AI Prediction
# --------------------------------------------------

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )


predicted_grade = classes[predicted.item()]

confidence_percentage = confidence.item() * 100


# --------------------------------------------------
# 6. Display Result
# --------------------------------------------------

print("\n========================================")
print("       TOMATO QUALITY RESULT")
print("========================================")

print(
    f"Predicted Grade : Grade {predicted_grade}"
)

print(
    f"Confidence      : {confidence_percentage:.2f}%"
)

print("----------------------------------------")

for i in range(3):

    print(
        f"Grade {classes[i]} : "
        f"{probabilities[0][i].item() * 100:.2f}%"
    )

print("========================================")