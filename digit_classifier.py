import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# ------------------------------
# Model Definition
# ------------------------------
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ------------------------------
# Training Setup
# ------------------------------
model = DigitClassifier()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# Training Loop
# ------------------------------
for epoch in range(3):  # increase epochs for better accuracy
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "digit_classifier.pth")
print("✅ Model saved!")

# ------------------------------
# Prediction on Custom Image
# ------------------------------
# Reload model (for demonstration)
model = DigitClassifier()
model.load_state_dict(torch.load("digit_classifier.pth"))
model.eval()

# Transform for input image
custom_transform = transforms.Compose([
    transforms.Grayscale(),         # ensure grayscale
    transforms.Resize((28, 28)),    # resize to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(img_path):
    img = Image.open(img_path)
    img = custom_transform(img).unsqueeze(0)   # add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Example usage
print("Predicted Digit:", predict_image("img0.jpg"))




# while using google colab : to upload images
# from google.colab import files
# uploaded = files.upload()
