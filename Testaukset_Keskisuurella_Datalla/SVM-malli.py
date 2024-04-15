import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from skimage.feature import hog
import time


# Funktio kuvien lataamiseen ja muuntamiseen
def load_images(directory):
    images = []
    labels = []
    for label in os.listdir(directory):
        image_count = 0
        for filename in os.listdir(os.path.join(directory, label)):
            img = cv2.imread(os.path.join(directory, label, filename))
            images.append(img.flatten())  # Flatten the inverted image
            labels.append(label)
            image_count += 1
            print("images", image_count, label)
    return images, labels

X, y = load_images('Muunnetut')

X = np.array(X)
y = np.array(y)

start_time = time.time()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
print("Mallin opetus alkaa")
svm_model = SVC(kernel='linear', C=2, gamma='auto')
svm_model.fit(X_train, y_train)
print("Mallin opetus loppuu")

# Evaluate the model
y_pred = svm_model.predict(X_test)

end_time = time.time()
elapsed_time = end_time - start_time

# Calculate accuracy, MSE, and R-squared
accuracy = accuracy_score(y_test, y_pred)

print("Elapsed time:", elapsed_time, "seconds")
print("Tarkkuus:", accuracy)

# Save the model
dump(svm_model, 'svm_model.pkl')
