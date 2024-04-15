import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
import joblib
import time

# Function to load images and labels from a directory
def load_images(directory):
    images = []
    labels = []
    for label in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, label)):
            img = cv2.imread(os.path.join(directory, label, filename))
            images.append(img.flatten())  # Flatten the inverted image
            labels.append(label)
    return images, labels

# Load images and labels
images, labels = load_images("Muunnetut")

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)
start_time = time.time()
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)
end_time = time.time()
elapsed_time = end_time - start_time
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save the model
joblib.dump(clf, 'decision_tree_model.pkl')
print("Elapsed time:", elapsed_time, "seconds")
# Print accuracy
print("Accuracy:", accuracy)
