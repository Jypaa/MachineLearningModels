import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import time

# Function to load data
def load_data(image_dir):
    X = []
    y = []
    class_labels = sorted(os.listdir(image_dir))
    for label in class_labels:
        label_dir = os.path.join(image_dir, label)
        image_count = 0
        # Map subfolder name to corresponding letter label
        letter_label = chr(int(label) + 64)  # Assuming letters A-Z correspond to subfolders 1-26
        for filename in os.listdir(label_dir):
            if image_count == 5000:  # Limiting to 30 images per class
                break
            image_path = os.path.join(label_dir, filename)
            # Load image
            img = imread(image_path)
            # Append to X and y lists
            X.append(img)
            y.append(letter_label)  # Use the mapped letter label
            image_count += 1
    return np.array(X), np.array(y)

# Load data
image_dir = 'Pieni'  # Update with your data directory
X, y = load_data(image_dir)
start_time = time.time()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Define input shape and number of classes
input_shape = X_train[0].shape
num_classes = len(np.unique(y))

# Create the CNN model
model = Sequential([
    # Adjust input shape to match image dimensions
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=512, validation_data=(X_test, y_test), callbacks=[early_stopping])

end_time = time.time()
elapsed_time = end_time - start_time

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.ylim(bottom=0, top=5 * min(history.history['val_loss']))
plt.show()

# Evaluate the model
test_accuracy = model.evaluate(X_test, y_test)
print("Elapsed time:", elapsed_time, "seconds")
print("Test Accuracy:", test_accuracy)

# Save the model
model.save("cnn_model.h5")
