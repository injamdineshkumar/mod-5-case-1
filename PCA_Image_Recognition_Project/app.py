
import streamlit as st
import numpy as np
import joblib

from PIL import Image, ImageOps
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Page configuration
st.set_page_config(
    page_title="PCA Image Recognition",
    page_icon="🖼️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("pca_digit_model.joblib")

model = load_model()

# Load model evaluation results
results = np.load("model_results.npz")

accuracy = float(results["accuracy"])
cm = results["confusion_matrix"]
variance = float(results["variance"])

# Application title
st.title("🖼️ PCA-Based Image Recognition")

st.write(
    "Image recognition using Principal Component Analysis "
    "(PCA) and Support Vector Machine (SVM)."
)

st.divider()

# Sidebar
st.sidebar.title("Project Information")

st.sidebar.info(
    """
    Project: Image Recognition Using PCA

    Feature Extraction: PCA

    Classification: SVM

    Dataset: Handwritten Digits

    Classes: 0 to 9
    """
)

# Model performance
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Original Features", "64")
col3.metric("PCA Components", "30")

st.write(
    f"Variance retained by PCA: {variance * 100:.2f}%"
)

st.divider()

# Image upload
st.subheader("Upload Handwritten Digit Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    # Show original image
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )

    # Convert white paper / dark ink into
    # the black background / bright strokes
    # convention of sklearn digits dataset.
    processed = ImageOps.autocontrast(image)
    processed = ImageOps.invert(processed)

    # Resize to 8 x 8 pixels
    processed = processed.resize(
        (8, 8),
        Image.Resampling.LANCZOS
    )

    # Convert pixel range from 0-255 to 0-16
    image_array = np.array(processed, dtype=np.float64)
    image_array = image_array * (16.0 / 255.0)

    # Flatten into 64 features
    image_features = image_array.reshape(1, -1)

    # Predict digit
    prediction = model.predict(image_features)[0]

    with col2:
        st.image(
            processed,
            caption="Processed 8 x 8 Image",
            width=200
        )

    st.success("Image recognition completed!")

    st.subheader("Prediction Result")

    st.markdown(
        f"<h1 style='text-align:center;'>"
        f"Predicted Digit: {prediction}"
        f"</h1>",
        unsafe_allow_html=True
    )

    st.write(
        "The uploaded image was converted into "
        "64 pixel features and processed by PCA "
        "before classification."
    )

else:
    st.info("Upload a handwritten digit image to begin.")

st.divider()

# PCA information
st.subheader("PCA Feature Extraction")

st.write(
    "PCA reduces the original 64 image features "
    "to 30 principal components."
)

st.progress(30 / 64)

st.write("Feature reduction: 53.125%")

# Confusion matrix
st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(8, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
).plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

st.pyplot(fig)
plt.close(fig)

st.divider()

st.subheader("Project Conclusion")

st.write(
    """
    PCA reduces image dimensionality and extracts
    important features. The SVM classifier uses
    these features to recognize handwritten digits.

    This application demonstrates feature extraction,
    dimensionality reduction and image classification
    using Python and Streamlit.
    """
)

st.caption("Developed using Python, Scikit-learn and Streamlit.")
