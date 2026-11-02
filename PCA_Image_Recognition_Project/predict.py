
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()

# Load trained model
model = joblib.load("pca_digit_model.joblib")

# Select an image
index = 25

image = digits.images[index]

# Convert image into feature vector
image_data = image.reshape(1, -1)

# Predict digit
prediction = model.predict(image_data)

# Display result
print("Predicted Digit:", prediction[0])
print("Actual Digit:", digits.target[index])

# Display image
plt.imshow(image, cmap="gray")
plt.title(f"Predicted Digit: {prediction[0]}")
plt.axis("off")
plt.show()
