import sys
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Class names
# -----------------------------
classes = ["fresh", "rotten"]

# -----------------------------
# 3. Load EfficientNet-B0
# -----------------------------
weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.load_state_dict(
    torch.load(
        "tomato_quality_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# -----------------------------
# 4. Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 5. Get image path
# -----------------------------
if len(sys.argv) < 2:
    print("Please provide an image path.")
    print("Example:")
    print(r'python src/predict.py "C:\Users\Dell\Pictures\tomato.jpg"')
    sys.exit()

image_path = sys.argv[1]

# -----------------------------
# 6. Open image
# -----------------------------
try:
    image = Image.open(image_path).convert("RGB")
except Exception as error:
    print("Could not open image.")
    print("Error:", error)
    sys.exit()

# -----------------------------
# 7. Prepare image
# -----------------------------
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(device)

# -----------------------------
# 8. Prediction
# -----------------------------
with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

# -----------------------------
# 9. Display result
# -----------------------------
prediction = classes[predicted_class.item()]
confidence_percentage = confidence.item() * 100

print("\n========== TOMATO QUALITY RESULT ==========")
print("Prediction :", prediction.upper())
print(f"Confidence : {confidence_percentage:.2f}%")
print("===========================================")