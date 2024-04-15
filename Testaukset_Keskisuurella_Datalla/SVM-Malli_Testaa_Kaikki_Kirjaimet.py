import os
import cv2
import joblib
import time
# Load the saved SVM model
loaded_model = joblib.load('svm_model.pkl')

# Define a mapping from class labels to letters
label_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
                   11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S',
                   20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}

# Path to the folder containing letter images
#folder_path = 'Testi'
folder_path = 'Kasinkirjoitettu'

# Initialize variables for accuracy calculation
total_images = 0
correct_predictions = 0
start_time = time.time()
# Iterate through each letter image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Assuming images are PNG or JPG
        # Load the letter image
        letter_img = cv2.imread(os.path.join(folder_path, filename))  # Read im-age in grayscale
        # Resize image to match the size used during training
        resized_img = cv2.resize(letter_img, (32, 32))
        # Make prediction using the loaded SVM model
        flattened_img = resized_img.flatten().reshape(1, -1)
        predicted_label = loaded_model.predict(flattened_img)
        # Get the actual label (letter) from the filename
        actual_label = os.path.splitext(filename)[0]
        # Increment the total number of images
        total_images += 1
        # Check if the prediction is correct
        try:
            predicted_letter = label_to_letter[int(predicted_label[0])]
        except KeyError:
            print("Predicted label:", predicted_label[0])
            print("Error: Predicted label does not exist in label_to_letter dictionary.")
            continue
        
        # Check if the prediction is correct
        if predicted_letter == actual_label:
            correct_predictions += 1
        print(actual_label, predicted_letter)

end_time = time.time()
elapsed_time = end_time - start_time

# Calculate accuracy
accuracy = (correct_predictions / total_images) * 100
print("Accuracy: {:.2f}%".format(accuracy))
print("Elapsed time:", elapsed_time, "seconds")
