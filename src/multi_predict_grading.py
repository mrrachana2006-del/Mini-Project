import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# --------------------------------------------------
# 1. Check image paths
# --------------------------------------------------

if len(sys.argv) != 4:
    print("Please provide exactly 3 tomato image paths.")
    print("Example:")
    print('python src/multi_predict_grading.py "image1.jpeg" "image2.jpeg" "image3.jpeg"')
    sys.exit()

image_paths = sys.argv[1:4]

# --------------------------------------------------
# 2. Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# 3. Load trained EfficientNet-B0 model
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

grades = ["A", "B", "C"]

all_predictions = []
all_probabilities = []

# --------------------------------------------------
# 5. Predict each image
# --------------------------------------------------

for image_path in image_paths:

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"\nCould not open: {image_path}")
        print("Error:", e)
        sys.exit()

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    predicted_grade = grades[prediction.item()]
    confidence_value = confidence.item() * 100

    all_predictions.append(prediction.item())
    all_probabilities.append(
        probabilities[0].cpu()
    )

    print("\n----------------------------------------")
    print("Image:", image_path)
    print("Prediction:", "Grade " + predicted_grade)
    print(f"Confidence: {confidence_value:.2f}%")

# --------------------------------------------------
# 6. Combine predictions
# --------------------------------------------------

grade_counts = {
    "A": 0,
    "B": 0,
    "C": 0
}

for prediction in all_predictions:

    grade = grades[prediction]

    grade_counts[grade] += 1

# Majority voting
final_grade = max(
    grade_counts,
    key=grade_counts.get
)

# --------------------------------------------------
# 7. Calculate average probabilities
# --------------------------------------------------

average_probabilities = torch.stack(
    all_probabilities
).mean(dim=0)

final_confidence = (
    average_probabilities[
        grades.index(final_grade)
    ].item() * 100
)

# --------------------------------------------------
# 8. Final result
# --------------------------------------------------

print("\n========================================")
print("       MULTI-VIEW TOMATO GRADING")
print("========================================")

print("Images analyzed :", len(image_paths))

print(
    f"Grade A views   : {grade_counts['A']}"
)

print(
    f"Grade B views   : {grade_counts['B']}"
)

print(
    f"Grade C views   : {grade_counts['C']}"
)

print("----------------------------------------")

print(
    "FINAL GRADE     : "
    + "Grade "
    + final_grade
)

print(
    f"Combined confidence: "
    f"{final_confidence:.2f}%"
)

print("========================================")