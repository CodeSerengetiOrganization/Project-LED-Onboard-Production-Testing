import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import pandas as pd
import numpy as np
import cv2  # Import OpenCV

# Load the trained model
model = tf.keras.models.load_model('resnet50_image_classifier.keras')

data_dir = 'images'

# Image dimensions
img_height, img_width = 100, 100
batch_size = 8

# Path to the test images folder
test_images_dir = 'test_images'

# Get a list of image files
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create lists to store results
filenames = []
predictions = []
confidences = []

# ImageDataGenerator for validation data (only rescaling)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create validation generator from directory
validation_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Manual class label mapping
class_labels = {v: k for k, v in validation_generator.class_indices.items()}

# Confidence threshold (adjust as needed)
confidence_threshold = 0.8

# Function to calculate color deviation using OpenCV
def calculate_color_deviation(image_path):
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_lime = np.array([30, 100, 100])
    upper_lime = np.array([90, 255, 255])

    mask = cv2.inRange(image_hsv, lower_lime, upper_lime)

    total_pixels = mask.shape[0] * mask.shape[1]
    non_lime_pixels = total_pixels - cv2.countNonZero(mask)
    deviation_percentage = (non_lime_pixels / total_pixels) * 100

    return deviation_percentage

# Loop through the images
for image_file in image_files:
    # Load and preprocess the image
    img_path = os.path.join(test_images_dir, image_file)
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Calculate color deviation
    deviation = calculate_color_deviation(img_path)
    deviation_array = np.array([[deviation]]) #create 2d array for the model.

    # Make a prediction (provide both inputs)
    prediction = model.predict([img_array, deviation_array])
    max_prob = np.max(prediction)
    predicted_class = np.argmax(prediction)

    if max_prob >= confidence_threshold:
        predicted_label = class_labels[predicted_class]
    else:
        predicted_label = "unknown"

    filenames.append(image_file)
    predictions.append(predicted_label)
    confidences.append(max_prob)

# Create a DataFrame and save to CSV
results_df = pd.DataFrame({'filename': filenames, 'prediction': predictions, 'confidence': confidences})
results_df.to_csv('test_results.csv', index=False)

print("Evaluation complete. Results saved to test_results.csv")