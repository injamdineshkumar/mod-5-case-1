
# PCA Image Recognition Using Streamlit

## Project Description
This project implements image recognition using
Principal Component Analysis (PCA) and Support
Vector Machine (SVM).

PCA reduces image dimensions and extracts important
features for efficient classification.

## Technologies
- Python
- Streamlit
- NumPy
- Scikit-learn
- Matplotlib
- Pillow
- Joblib

## Dataset
Scikit-learn Handwritten Digits Dataset.

## Features
1. Image upload
2. Image preprocessing
3. PCA feature extraction
4. SVM classification
5. Digit prediction
6. Accuracy calculation
7. Confusion matrix visualization

## How to Run

From the project root, install dependencies, then run the training script and app from the project folder:

```powershell
python -m pip install -r requirements.txt
Set-Location .\PCA_Image_Recognition_Project
python train.py
python -m streamlit run app.py
```

## Conclusion
The application demonstrates how PCA can reduce
image dimensions and extract important features
for machine learning-based image recognition.
