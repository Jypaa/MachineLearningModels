import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib
import time

# Define constants
num_classes = 26 

# Load and preprocess the data
X = []
y = []
for i in range(1, num_classes + 1):
    class_dir = os.path.join('Isot', str(i))
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path)
        X.append(image.flatten())
        y.append(i)

# Convert lists to arrays
X = np.array(X)
y = np.array(y)

start_time = time.time()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN model
k = 1  # Number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn_model.predict(X_test)

end_time = time.time()
elapsed_time = end_time - start_time

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Elapsed time:", elapsed_time, "seconds")
print("Accuracy:", accuracy)

# Save the model
joblib.dump(knn_model, 'knn_model.pkl')
