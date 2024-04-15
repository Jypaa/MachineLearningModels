import os
import cv2
import joblib
import time

# Load the saved model
loaded_model = joblib.load('decision_tree_model.pkl')

# Function to preprocess a single letter image
def preprocess_image(img):
    # Resize image to match the size used during training
    img = cv2.resize(img, (32, 32))
    # Flatten image
    img = img.reshape(1, -1)
    return img

# Define a mapping from class labels to letters
label_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
                   11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S',
                   20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}

# Path to the folder containing letter images
#folder_path = 'Testi'  # Adjust the path as needed
folder_path = 'Kasinkirjoitettu'

# Initialize variables for accuracy calculation
total_images = 0
correct_predictions = 0

start_time = time.time()

# Iterate through each letter image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Assuming images are PNG or JPG
        # Load the letter image
        letter_img = cv2.imread(os.path.join(folder_path, filename))
        # Preprocess the letter image
        preprocessed_img = preprocess_image(letter_img)
        # Make prediction using the loaded model
        predicted_label = loaded_model.predict(preprocessed_img)
        # Get the actual label (letter) from the filename
        actual_label = os.path.splitext(filename)[0]
        # Increment the total number of images
        total_images += 1
        # Check if the prediction is correct
        if label_to_letter[int(predicted_label[0])] == actual_label:
            correct_predictions += 1
        print(actual_label, label_to_letter[int(predicted_label[0])])
# Calculate accuracy

end_time = time.time()
elapsed_time = end_time - start_time

accuracy = (correct_predictions / total_images) * 100
print("Accuracy: {:.2f}%".format(accuracy))
print("Elapsed time:", elapsed_time, "seconds")
