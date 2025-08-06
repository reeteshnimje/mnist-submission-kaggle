import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# ----------------------------
# 1. LOAD AND PREPROCESS DATA
# ----------------------------
train_df = pd.read_csv("digit-recognizer/train.csv")
test_df = pd.read_csv("digit-recognizer/test.csv")

# Split features (X) and labels (y)
X = train_df.iloc[:, 1:].values.astype('float32')  # pixel columns
y = train_df['label'].values.astype('int64')       # labels

# Normalize: scale pixels to [0,1] then standardize
X = X / 255.0
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8
X = (X - mean) / std

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_val_tensor   = torch.tensor(X_val)
y_val_tensor   = torch.tensor(y_val)

# Test set: normalize using SAME mean/std as training
X_test = test_df.values.astype('float32')
X_test = X_test / 255.0
X_test = (X_test - mean) / std
X_test_tensor = torch.tensor(X_test)

# Dataloaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_tensor), batch_size=64, shuffle=False)

# ----------------------------
# 2. DEFINE MODEL
# ----------------------------
class HugeNet(nn.Module):
    def __init__(self):
        super(HugeNet, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        self.bn1 = nn.BatchNorm1d(32)

        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)

        self.fc_out = nn.Linear(16, 10)

        self.act = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.001)

        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.dropout(self.act(self.bn1(self.fc1(x))))
        x = self.dropout(self.act(self.bn2(self.fc2(x))))
        x = self.dropout(self.act(self.bn3(self.fc3(x))))
        x = self.dropout(self.act(self.bn4(self.fc4(x))))
        x = self.fc_out(x)  # logits (raw scores)
        return x

# Instantiate model
model = HugeNet()

# ----------------------------
# 3. TRAINING SETUP
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

# ----------------------------
# 4. TRAINING LOOP
# ----------------------------
epochs = 200
best_val_acc = 0.0

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    scheduler.step(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")

# ----------------------------
# 5. PREDICT TEST SET (KAGGLE)
# ----------------------------
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

predictions = []
with torch.no_grad():
    for (inputs,) in test_loader:  # Note the comma! (single-element tuple)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())

# Save submission file
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(predictions)+1),
    "Label": predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
