import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Load dataset
df = pd.read_csv("data/credit_risk_dataset.csv")

# Clean data
df = df.drop_duplicates()
df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"] <= 50]

# Separate features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Identify columns
categorical_columns = X.select_dtypes(include=["object", "string"]).columns
numerical_columns = X.select_dtypes(exclude=["object", "string"]).columns

# One-hot encoding
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_categorical = encoder.fit_transform(
    X[categorical_columns]
)

# Scale numerical features
scaler = StandardScaler()

X_numerical = scaler.fit_transform(
    X[numerical_columns]
)

# Save preprocessing objects
joblib.dump(
    encoder,
    "models/pytorch_encoder.pkl"
)

joblib.dump(
    scaler,
    "models/pytorch_scaler.pkl"
)

# Combine features
X_final = np.hstack([
    X_numerical,
    X_categorical
])

# Check and clean invalid numerical values
X_final = np.nan_to_num(
    X_final,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert to tensors
X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)


# PyTorch Neural Network
class LoanRiskModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# Create model
model = LoanRiskModel(
    X_train.shape[1]
)

# Calculate class weight for imbalanced data
positive_count = y_train.sum()
negative_count = len(y_train) - positive_count

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
epochs = 50

for epoch in range(epochs):

    model.train()

    logits = model(X_train_tensor)

    loss = criterion(
        logits,
        y_train_tensor
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )


# Evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

model.eval()

with torch.no_grad():

    logits = model(X_test_tensor)

    probabilities = torch.sigmoid(logits)

    predicted_classes = (
        probabilities >= 0.5
    ).int().numpy().flatten()

    actual_classes = (
        y_test_tensor.int().numpy().flatten()
    )

    probability_values = (
        probabilities.numpy().flatten()
    )


# Classification report
print("\nClassification Report:")
print(
    classification_report(
        actual_classes,
        predicted_classes
    )
)


# Confusion matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        actual_classes,
        predicted_classes
    )
)


# ROC-AUC
roc_auc = roc_auc_score(
    actual_classes,
    probability_values
)

print(
    f"\nPyTorch ROC-AUC: {roc_auc:.4f}"
)


# Accuracy
accuracy = (
    predicted_classes == actual_classes
).mean()

print(
    f"PyTorch Test Accuracy: {accuracy:.4f}"
)


# Save model
torch.save(
    model.state_dict(),
    "models/pytorch_loan_risk_model.pth"
)

print(
    "\nPyTorch model saved successfully."
)