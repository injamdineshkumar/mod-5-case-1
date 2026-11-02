
import joblib
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load handwritten digit dataset
digits = load_digits()

X = digits.data
y = digits.target

print("Dataset loaded:", X.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# PCA and SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=30)),
    ("svm", SVC(kernel="rbf", C=5))
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Training completed!")
print("Model Accuracy:", accuracy * 100, "%")

# Save model
joblib.dump(model, "pca_digit_model.joblib")

# Save evaluation information
np.savez(
    "model_results.npz",
    accuracy=accuracy,
    confusion_matrix=confusion_matrix(y_test, predictions),
    variance=model.named_steps["pca"].explained_variance_ratio_.sum()
)

print("Model saved successfully!")
