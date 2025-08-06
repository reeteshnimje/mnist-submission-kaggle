# ----------------------------
# IMPORT NECESSARY LIBRARIES
# ----------------------------
import torch                   # PyTorch core library (for tensors, neural nets)
import torch.nn as nn          # nn = "neural network" module (defines layers, activations, etc.)
import torch.optim as optim    # Optimizers (e.g., SGD, Adam) to update weights
import pandas as pd            # Pandas is for reading CSV files and basic data manipulation
from sklearn.model_selection import train_test_split  # To split training data into train & validation sets
from torch.utils.data import DataLoader, TensorDataset # Converts tensors into iterable batches for training

# ----------------------------
# STEP 1: LOAD THE DATA FROM CSV
# ----------------------------

# Load training data (has labels in the first column)
train_df = pd.read_csv("digit-recognizer/train.csv")

# Load test data (no labels, just pixel values)
test_df = pd.read_csv("digit-recognizer/test.csv")

# Extract labels (the digit classes: 0-9) from training CSV
y = train_df["label"].values   # Numpy array of shape (num_samples,)
# Extract pixel values (784 columns: pixel0, pixel1, ..., pixel783) from training CSV
X = train_df.drop("label", axis=1).values  # Shape: (num_samples, 784)

# Convert pixel values from [0, 255] to [0, 1] for numerical stability
X = X / 255.0
test_data = test_df.values / 255.0  # Apply the same scaling to test data

# ----------------------------
# STEP 2: SPLIT INTO TRAIN AND VALIDATION SETS
# ----------------------------
# We'll keep aside 20% of training data as validation (to measure generalization)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# STEP 3: CONVERT DATA INTO TORCH TENSORS
# ----------------------------
# PyTorch uses its own tensor type, similar to numpy arrays but optimized for deep learning.
# Convert numpy arrays into float tensors (for inputs) and long tensors (for class labels)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)   # Shape: [num_train_samples, 784]
y_train_tensor = torch.tensor(y_train, dtype=torch.long)      # Shape: [num_train_samples]

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(test_data, dtype=torch.float32)  # Test data has no labels

# ----------------------------
# STEP 4: CREATE DATALOADERS
# ----------------------------
# Dataloaders allow you to efficiently batch the data and shuffle it during training.
batch_size = 64  # Number of samples per training step

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# STEP 5: DEFINE A SIMPLE NEURAL NETWORK
# ----------------------------
# We will build a VERY simple network:
# Input: 784 features (pixels)
# Hidden layer: 3 neurons (very small to keep it basic)
# Output: 10 neurons (one for each digit 0-9)

# REPLACE TinyNet CLASS WITH THIS:
class BiggerNet(nn.Module):
    def __init__(self):
        super(BiggerNet, self).__init__()
        # Layer 1: Fully connected, 784 inputs → 16 neurons
        self.fc1 = nn.Linear(784, 16)
        # Layer 2: Fully connected, 16 → 32 neurons
        self.fc2 = nn.Linear(16, 32)
        # Layer 3: Fully connected, 32 → 16 neurons
        self.fc3 = nn.Linear(32, 16)
        # Output layer: 16 → 10 (digits 0-9)
        self.fc_out = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))      # Layer 1 + ReLU
        x = torch.relu(self.fc2(x))      # Layer 2 + ReLU
        x = torch.relu(self.fc3(x))      # Layer 3 + ReLU
        x = self.fc_out(x)               # Final layer (raw logits for classification)
        return x

# Instantiate new model
model = BiggerNet()


# ----------------------------
# STEP 6: DEFINE LOSS FUNCTION AND OPTIMIZER
# ----------------------------
# Loss function: CrossEntropyLoss is standard for classification problems.
criterion = nn.CrossEntropyLoss()

# Optimizer: We'll use Stochastic Gradient Descent (SGD) with learning rate 0.1
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ----------------------------
# STEP 7: TRAINING LOOP
# ----------------------------
num_epochs = 250  # We'll train for 5 passes over the training data

for epoch in range(num_epochs):
    model.train()  # Set the model in "training mode"
    running_loss = 0.0

    for images, labels in train_loader:  # Loop over batches
        optimizer.zero_grad()            # Reset gradients to zero (otherwise they accumulate)
        outputs = model(images)          # Forward pass: compute predictions
        loss = criterion(outputs, labels)# Compute loss between predictions & true labels
        loss.backward()                  # Backpropagation: compute gradients of loss wrt weights
        optimizer.step()                 # Update weights using gradients

        running_loss += loss.item()      # Track cumulative loss for reporting

    # Compute average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

    # ----------------------------
    # VALIDATION STEP (AFTER EACH EPOCH)
    # ----------------------------
    model.eval()  # Set model to "evaluation mode" (disables dropout/batchnorm if any)
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Pick class with max logit score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc*100:.2f}%")

# ----------------------------
# STEP 8: PREDICT ON TEST DATA
# ----------------------------
model.eval()  # Final evaluation mode
test_loader = DataLoader(X_test_tensor, batch_size=batch_size, shuffle=False)

all_preds = []
with torch.no_grad():
    for images in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.tolist())  # Convert tensor predictions to Python list

# ----------------------------
# STEP 9: CREATE SUBMISSION FILE
# ----------------------------
submission_df = pd.DataFrame({
    "ImageId": range(1, len(all_preds) + 1),
    "Label": all_preds
})

submission_df.to_csv("submission-bignet.csv", index=False)
print("submission.csv created successfully!")
