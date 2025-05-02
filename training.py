import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision.datasets import ImageFolder




# --- Hyperparameters ---
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 10
n_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Transforms ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])


# Correct dataset loading
dataset = ImageFolder(root='D:\\Plant ML project\\dataset', transform=transform)
class_names = dataset.classes

# --- Split Dataset ---
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# --- Define Model ---
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear((IMAGE_SIZE//8)*(IMAGE_SIZE//8)*64, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

model = CNNModel().to(device)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total = 0
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss_list.append(train_loss/len(train_loader))
    train_acc_list.append(100 * correct / total)

    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss_list.append(val_loss/len(val_loader))
    val_acc_list.append(100 * correct / total)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss_list[-1]:.4f}, Val Loss: {val_loss_list[-1]:.4f}")

# --- Plot Accuracy & Loss ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()

# --- Save Model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/plant_model.pth")
def predict_image(image, model):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return class_names[predicted.item()]

# Show Predictions
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        plt.figure(figsize=(12, 6))
        for i in range(6):
            ax = plt.subplot(2, 3, i+1)
            
            img = transforms.ToPILImage()(images[i].cpu())
            pred_class = predict_image(img, model)
            actual = class_names[labels[i].item()]
            
            plt.imshow(img)
            plt.title(f"Actual: {actual}\nPredicted: {pred_class}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        break


# # Show Predictions
# for images, labels in test_loader:
#     plt.figure(figsize=(12, 6))
#     for i in range(6):
#         ax = plt.subplot(2, 3, i+1)
#         img = transforms.ToPILImage()(images[i])
#         pred_class = predict_image(img, model)
#         actual = class_names[labels[i]]
#         plt.imshow(img)
#         plt.title(f"Actual: {actual}\nPredicted: {pred_class}")
#         plt.axis("off")
#     break
