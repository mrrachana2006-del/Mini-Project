import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# --------------------------------------------------
# 1. Check image path
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Please provide the tomato image path.")
    print("Example:")
    print('python src/predict_grading.py "C:\\path\\to\\tomato.jpg"')
    sys.exit()

image_path = sys.argv[1]

# --------------------------------------------------
# 2. Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 3. Load EfficientNet-B0
# --------------------------------------------------

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
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
# 4. Image transformation
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# 5. Load image
# --------------------------------------------------

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print("Could not open image.")
    print("Error:", e)
    sys.exit()

image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

# --------------------------------------------------
# 6. Prediction
# --------------------------------------------------

with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(
        probabilities,
        1
    )

# --------------------------------------------------
# 7. Convert prediction to grade
# --------------------------------------------------

grades = ["A", "B", "C"]

predicted_grade = grades[prediction.item()]
confidence_percentage = confidence.item() * 100

# --------------------------------------------------
# 8. Display result
# --------------------------------------------------

print("\n========================================")
print("       TOMATO QUALITY GRADING")
print("========================================")

print("Prediction :", "Grade " + predicted_grade)
print(f"Confidence : {confidence_percentage:.2f}%")

print("----------------------------------------")

print(
    f"Grade A probability: "
    f"{probabilities[0][0].item() * 100:.2f}%"
)

print(
    f"Grade B probability: "
    f"{probabilities[0][1].item() * 100:.2f}%"
)

print(
    f"Grade C probability: "
    f"{probabilities[0][2].item() * 100:.2f}%"
)

print("========================================")