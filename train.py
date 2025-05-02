# Set seeds for reproducibility
import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Other imports
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

# Dataset Path
base_dir = r'D:\\Plant ML project\\dataset'

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Dataset
dataset = datasets.ImageFolder(root=base_dir, transform=transform)
class_names = dataset.classes
print("Classes:", class_names)

# Splitting dataset into train and validation
val_split = 0.2
num_total = len(dataset)
num_val = int(val_split * num_total)
num_train = num_total - num_val
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

# Data Loaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model Definition
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(64 * 56 * 56, 256),
    nn.ReLU(),
    nn.Linear(256, len(class_names))
)

model = model.to(device)
print(model)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 5
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = correct / total

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Plot training & validation loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save class indices (class names)
class_indices = {v: k for k, v in dataset.class_to_idx.items()}
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

# Function to Load and Preprocess Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

# Predict function
def predict_image_class(model, image_path, class_indices):
    model.eval()
    img_tensor = load_and_preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()
        predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

# Example Usage (Optional)
test_image_path = r'D:\\Plant ML project\dataset\\Potato___healthy\\0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG  # <-- Change this to a valid image if you want to test'

if os.path.exists(test_image_path):
    predicted_class_name = predict_image_class(model, test_image_path, class_indices)
    print("Predicted Class Name:", predicted_class_name)
else:
    print("No test image provided. Skipping prediction.")

# Save the trained model
torch.save(model.state_dict(), 'plant_disease_model.pth')
print("Model saved successfully!")
