import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Paths
DATA_PATH = os.path.join("dataset", "user_purchase_dataset.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    # Load dataset
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please ensure the file exists.")
    exit()

# Separate features (X) and target (y)
# We drop UserID as it is not a predictive feature
X = df.drop(["UserID", "Purchased"], axis=1)
y = df["Purchased"]

# Since SVM is sensitive to feature scaling, we will use a StandardScaler.
# We'll use a pipeline to combine scaling and the SVM model for a cleaner process.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42, probability=True))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model pipeline
pipeline.fit(X_train, y_train)

# Save the trained pipeline to a .pkl file
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"Model saved to {MODEL_PATH}")

