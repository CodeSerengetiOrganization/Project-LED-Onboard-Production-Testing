import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import pandas as pd
import numpy as np
import cv2  # Import OpenCV

# Load the trained model
model = tf.keras.models.load_model('resnet50_image_classifier.keras')  # Load .keras model
# model = tf.keras.models.load_model('VGG16_image_classifier.keras')  # Load .keras model

data_dir = 'images'

# Image dimensions
img_height, img_width = 100, 100
batch_size = 8

# Path to the test images folder
test_images_dir = 'test_images'  # Corrected path

# Get a list of image files
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create lists to store results
filenames = []
predictions = []
confidences = []  # add confidence level.
color_deviation_percentage = []

# ImageDataGenerator for validation data (only rescaling)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# Create validation generator from directory
validation_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Very important to set shuffle to false.
)

# Manual class label mapping
class_labels = {v: k for k, v in validation_generator.class_indices.items()}

# Confidence threshold (adjust as needed)
confidence_threshold = 0.6

# Color check parameters (adjust as needed)
lime_hue_range = (86, 95)
lime_saturation_range = (200, 255)
lime_value_range = (170, 255)
color_deviation_percentage_threshold = 80  # Adjust as needed

def predict_with_color_check(image_path, model, lime_hue_range, lime_saturation_range, lime_value_range, color_deviation_percentage_threshold, confidence_threshold):
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_lime = np.array([lime_hue_range[0], lime_saturation_range[0], lime_value_range[0]])
    upper_lime = np.array([lime_hue_range[1], lime_saturation_range[1], lime_value_range[1]])

    mask = cv2.inRange(image_hsv, lower_lime, upper_lime)

    total_pixels = mask.shape[0] * mask.shape[1]
    non_lime_pixels = total_pixels - cv2.countNonZero(mask)
    deviation_percentage = (non_lime_pixels / total_pixels) * 100
    color_deviation_percentage.append(deviation_percentage)

    print(f"Image: {image_path}, Deviation Percentage: {deviation_percentage:.2f}%")

    if deviation_percentage > color_deviation_percentage_threshold:
        return "unknown", 0.0
    else:
        # Use the already loaded image (from cv2)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert from BGR to RGB
        image_resized = cv2.resize(image_rgb, (100, 100)) #resize the image.
        img_array = image_resized / 255.0 #rescale the image.
        img_array = np.expand_dims(img_array, axis=0) #expand the dimensions.

        prediction = model.predict(img_array)
        max_prob = np.max(prediction)
        predicted_class = np.argmax(prediction)

        if max_prob >= confidence_threshold:
            predicted_label = class_labels[predicted_class]
            return predicted_label, max_prob
        else:
            return "unknown", max_prob

# Loop through the images
for image_file in image_files:
    img_path = os.path.join(test_images_dir, image_file)
    predicted_label, max_prob = predict_with_color_check(img_path, model, lime_hue_range, lime_saturation_range, lime_value_range, color_deviation_percentage_threshold, confidence_threshold)

    filenames.append(image_file)
    predictions.append(predicted_label)
    confidences.append(max_prob)
    

# Create a DataFrame and save to CSV
results_df = pd.DataFrame({'filename': filenames, 'prediction': predictions, 'confidence': confidences,'color_deviation_percentage':color_deviation_percentage})
results_df.to_csv('test_results.csv', index=False)

# Append the parameters to a separate file or to the same file as metadata.
with open('test_results_parameters.txt', 'w') as param_file:
    param_file.write(f"lime_hue_range: {lime_hue_range}\n")
    param_file.write(f"lime_saturation_range: {lime_saturation_range}\n")
    param_file.write(f"lime_value_range: {lime_value_range}\n")
    param_file.write(f" confidence threshold: {confidence_threshold}\n")


print("Evaluation complete. Results saved to test_results.csv")