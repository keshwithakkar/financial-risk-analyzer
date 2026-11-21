import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

print("pytorch_predict.py started")

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Neural Network
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


# Load preprocessing
encoder = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "pytorch_encoder.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "pytorch_scaler.pkl"
    )
)


# Load training data to identify columns
df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "credit_risk_dataset.csv"
    )
)

df = df.drop_duplicates()

df = df[
    (df["person_age"] <= 100) &
    (df["person_emp_length"] <= 50)
]

X = df.drop("loan_status", axis=1)

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["object", "string"]
).columns


# Load model
sample_input = np.zeros(
    len(numerical_columns) +
    len(
        encoder.get_feature_names_out(
            categorical_columns
        )
    )
)

model = LoanRiskModel(
    len(sample_input)
)

model.load_state_dict(
    torch.load(
        os.path.join(
            BASE_DIR,
            "models",
            "pytorch_loan_risk_model.pth"
        ),
        map_location="cpu"
    )
)

model.eval()


def predict_risk(input_data):

    input_df = pd.DataFrame(
        [input_data]
    )

    # Encode categorical features
    categorical_data = encoder.transform(
        input_df[categorical_columns]
    )

    # Scale numerical features
    numerical_data = scaler.transform(
        input_df[numerical_columns]
    )

    # Combine
    final_features = np.hstack([
        numerical_data,
        categorical_data
    ])

    final_features = np.nan_to_num(
        final_features
    )

    # Convert to tensor
    tensor = torch.tensor(
        final_features,
        dtype=torch.float32
    )

    # Prediction
    with torch.no_grad():

        logits = model(tensor)

        probability = torch.sigmoid(
            logits
        ).item()

    risk = (
        "High Risk"
        if probability >= 0.5
        else "Low Risk"
    )

    return {
        "risk": risk,
        "probability": probability
    }

if __name__ == "__main__":

    test_data = {
        "person_age": 25,
        "person_income": 30000,
        "person_home_ownership": "RENT",
        "person_emp_length": 2,
        "loan_intent": "PERSONAL",
        "loan_grade": "C",
        "loan_amnt": 10000,
        "loan_int_rate": 12.0,
        "loan_percent_income": 0.33,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }

    result = predict_risk(test_data)

    print("\nPyTorch Prediction:")
    print(result)