import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the saved CNN model
loaded_model = load_model('cnn_model.h5')

# Function to preprocess a single image
def preprocess_image(img):
    # Resize image to match the size used during training (100x100 for this example)
    img = cv2.resize(img, (32, 32))
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    # Add a channel dimension
    img = np.expand_dims(img, axis=-1)
    return img

# Define a mapping from class labels to letters starting from 1
label_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                   10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                   19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Path to the folder containing letter images
#folder_path = 'Testi'
folder_path = 'Kasinkirjoitettu'

# Initialize variables for accuracy calculation
total_images = 0
correct_predictions = 0
letters =[]
start_time = time.time()
# Iterate through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Assuming images are PNG or JPG
        # Load the image
        img = cv2.imread(os.path.join(folder_path, filename))
        # Preprocess the image
        preprocessed_img = preprocess_image(img)
        # Make prediction using the loaded CNN model
        predicted_label = np.argmax(loaded_model.predict(preprocessed_img), axis=1)
        # Get the actual label from the filename
        actual_label = os.path.splitext(filename)[0]
        # Increment the total number of images
        total_images += 1
        # Check if the prediction is correct
        if label_to_letter[int(predicted_label[0])] == actual_label:
            correct_predictions += 1
        letters.append((actual_label, label_to_letter[int(predicted_label[0])]))
end_time = time.time()
elapsed_time = end_time - start_time
for actual_label, predicted_label in letters:
    print(actual_label, predicted_label)
    
# Calculate accuracy
accuracy = (correct_predictions / total_images) * 100
print("Accuracy: {:.2f}%".format(accuracy))
print("Elapsed time:", elapsed_time, "seconds")