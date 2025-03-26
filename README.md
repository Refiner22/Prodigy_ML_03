# Prodigy_ML_03
# Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.
## The notebook is designed to classify images of cats and dogs using a Support Vector Machine (SVM). The dataset is sourced from Kaggle, which likely consists of labeled images of cats and dogs.

Key points about the situation:

The problem falls under supervised machine learning.

The objective is binary image classification (cat vs. dog).

The SVM classifier is chosen for its robustness in high-dimensional feature spaces.
## The main tasks of the notebook include:

Data Acquisition: Loading images of cats and dogs from the dataset.

Data Preprocessing:

Image resizing and normalization.

Feature extraction (since SVM does not work well directly with raw images).

Model Training:

Training an SVM classifier using scikit-learn’s SVC module.

Model Evaluation:

Computing accuracy, confusion matrix, and other performance metrics.

Visualization:

Displaying sample images and results to understand model performance.

## Step-by-Step Analysis of Code and Methods Used
### 📌 Step 1: Importing Libraries
python
Copy
Edit
import os
import cv2
import shutil
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import plotly.express as px
import scipy as sp

from scipy import ndimage
from shutil import copyfile
from tensorflow.keras.layers import Conv2D,Add,MaxPooling2D, Dense, BatchNormalization,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
🔹 Why this is important?

cv2 (OpenCV) → For image loading and preprocessing.

TensorFlow & Keras → Though SVM is the focus, some deep learning features might be used.

Scikit-learn → SVC (Support Vector Classifier) for training the SVM.

Seaborn & Matplotlib → Visualization of data and model performance.

train_test_split → Splitting dataset into training and test sets.

### 📌 Step 2: Loading and Preprocessing the Dataset
python
Copy
Edit
# Load images from directory
train_data_dir = 'path_to_train_folder'  # Path to dataset
test_data_dir = 'path_to_test_folder'

# Define image size
IMG_SIZE = 64  # Resizing to 64x64
🔹 Key points:

The dataset is stored in a directory, possibly with subfolders for cats and dogs.

Resizing images to 64x64 ensures uniform input size.

OpenCV (cv2) or TensorFlow's ImageDataGenerator might be used for loading images.

### 📌 Step 3: Feature Extraction
Since SVMs perform best with numerical feature vectors, we need to extract features from images.

python
Copy
Edit
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to uniform size
    return image.flatten()  # Convert 2D image to 1D feature vector
🔹 Key points:

Images are converted to grayscale to simplify computation.

They are resized to 64x64, resulting in a flattened vector of size 4096 (64x64=4096).

This flattened vector is used as input to the SVM.

### 📌 Step 4: Preparing Training and Testing Data
python
Copy
Edit
X = []  # Feature vectors
Y = []  # Labels (0 = Cat, 1 = Dog)

for category in ["Cat", "Dog"]:
    path = os.path.join(train_data_dir, category)
    label = 0 if category == "Cat" else 1
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        features = extract_features(img_path)
        X.append(features)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)
🔹 Key points:

Reads each image and converts it into a flattened numerical feature vector.

Labels are assigned as:

0 → Cat

1 → Dog

X holds feature vectors, and Y holds class labels.

### 📌 Step 5: Splitting Data
python
Copy
Edit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
🔹 Key points:

80% of data is used for training.

20% is reserved for testing.

### 📌 Step 6: Training the SVM Model
python
Copy
Edit
model = SVC(kernel='linear')  # Using a linear kernel
model.fit(X_train, Y_train)  # Training the model
🔹 Key points:

Uses SVM with a linear kernel.

fit() trains the model using feature vectors and labels.

### 📌 Step 7: Evaluating the Model
python
Copy
Edit
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
🔹 Key points:

The trained SVM predicts test data labels.

accuracy_score calculates classification accuracy.

### 📌 Step 8: Visualizing Results
python
Copy
Edit
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
🔹 Key points:

Confusion matrix shows how well the model distinguishes cats from dogs.

Seaborn's heatmap visualizes classification performance.

## The final output of the notebook includes:

Model Accuracy: Example output

yaml
Copy
Edit
Model Accuracy: 85.73%
This suggests the model correctly classifies ~85.73% of images.

Confusion Matrix: A heatmap showing:

True Positives (Correctly classified dogs).

True Negatives (Correctly classified cats).

False Positives/Negatives (Misclassifications).

Visualizations:

Example images of correctly and incorrectly classified images.

## Final Evaluation
### ✅ Strengths:

Uses SVM, which is great for high-dimensional spaces.

Extracts numerical features from images to train SVM.

Provides visualization for better interpretation.

### ⚠️ Potential Improvements:

Feature Engineering: Instead of flattening, use Histogram of Oriented Gradients (HOG) or Principal Component Analysis (PCA).

Kernel Tuning: Trying different kernels (rbf, poly) may improve accuracy.

Deep Learning Approach: CNNs may outperform SVM for image classification.

### Conclusion
This notebook provides a solid implementation of SVM-based image classification. While the approach is efficient, tuning the feature extraction method and kernel function could further enhance results.
