import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# ------------------------
# Model Definition
# ------------------------
class DigitGeneratorNet(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(DigitGeneratorNet, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        x = torch.cat([noise, c], dim=1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

# ------------------------
# Training Setup
# ------------------------

# Transform and load MNIST dataset (not used in training directly but for structure)
transform = transforms.ToTensor()
train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss, optimizer
model = DigitGeneratorNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# ------------------------
# Training Loop
# ------------------------
print("Training started...")
for epoch in range(10):  # You can increase for better quality
    running_loss = 0.0
    for _ in range(len(train_loader)):
        # Generate random noise and labels
        labels = torch.randint(0, 10, (64,))
        noise = torch.randn(64, 100)

        # Generate fake images
        fake_images = model(noise, labels)

        # Use ideal target of all ones for simplicity
        target = torch.ones_like(fake_images)

        loss = criterion(fake_images, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/10], Loss: {running_loss:.4f}")

# ------------------------
# Save Trained Model
# ------------------------
torch.save(model.state_dict(), "digit_generator.pth")
print("Training complete. Model saved to 'digit_generator.pth'.")
