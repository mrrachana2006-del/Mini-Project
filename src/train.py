import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -----------------------------
# 2. Dataset paths
# -----------------------------
train_dir = "dataset/tomato/train"
validation_dir = "dataset/tomato/validation"

# -----------------------------
# 3. Image preprocessing
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 4. Load datasets
# -----------------------------
train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    validation_dir,
    transform=validation_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))

# -----------------------------
# 5. DataLoaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# 6. Load EfficientNet-B0
# -----------------------------
weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)

# Freeze the pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model = model.to(device)

# -----------------------------
# 7. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier[1].parameters(),
    lr=0.001
)

# -----------------------------
# 8. Training
# -----------------------------
epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()

    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            validation_total += labels.size(0)
            validation_correct += (
                predicted == labels
            ).sum().item()

    validation_accuracy = (
        100 * validation_correct / validation_total
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )

# -----------------------------
# 9. Save model
# -----------------------------
torch.save(
    model.state_dict(),
    "tomato_quality_model.pth"
)

print("\nTraining completed!")
print("Model saved as tomato_quality_model.pth")