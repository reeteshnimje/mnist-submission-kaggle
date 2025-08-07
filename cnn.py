import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitRecognizerDataset(Dataset):
    def __init__(self, dataframe, transform=None, train=True):
        self.data = dataframe
        self.transform = transform
        self.train = train

        if self.train:
            self.labels = self.data['label'].values
            self.images = self.data.drop(columns=['label']).values
        else:
            self.labels = None
            self.images = self.data.values

        self.images = self.images.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        if self.train:
            label = self.labels[idx]
            return image, label
        return image


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Load full train/test data once
full_train_df = pd.read_csv('digit-recognizer/train.csv')
test_df = pd.read_csv('digit-recognizer/test.csv')

test_dataset = DigitRecognizerDataset(test_df, transform=test_transform, train=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_accuracy = 0.0

for fold, (train_idx, val_idx) in enumerate(kfold.split(full_train_df)):
    print(f"\n--- Fold {fold + 1} ---")

    train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = DigitRecognizerDataset(train_df, transform=train_transform, train=True)
    val_dataset = DigitRecognizerDataset(val_df, transform=test_transform, train=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model  # Save model for final test prediction

# Final test prediction using best model
def predict_test(model):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv('submission-cnn.csv', index=False)
    print("\nPredictions saved to submission-cnn.csv")

predict_test(best_model)
