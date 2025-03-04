import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import pandas as pd
import numpy as np



# Load the trained model
model = tf.keras.models.load_model('resnet50_image_classifier.keras')  # Load .keras model

data_dir = 'images'

# Image dimensions
img_height, img_width = 100, 100
batch_size = 8

# Path to the test images folder
test_images_dir = 'test_images' # Corrected path

# Get a list of image files
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create lists to store results
filenames = []
predictions = []
confidences = [] #add confidence level.

# ImageDataGenerator for validation data (only rescaling)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create validation generator from directory
validation_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False #Very important to set shuffle to false.
)

# Manual class label mapping
# class_labels = {0: 'pass', 1: 'led_missing'}  # Replace with your actual class names. Make sure it matches the training data.
class_labels = {v: k for k, v in validation_generator.class_indices.items()}

# Confidence threshold (adjust as needed)
confidence_threshold = 0.8

# Loop through the images
for image_file in image_files:
    # Load and preprocess the image
    img_path = os.path.join(test_images_dir, image_file)
    img = image.load_img(img_path, target_size=(100, 100))  # Ensure target size matches training.
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Ensure rescaling matches training.

    # Make a prediction
    prediction = model.predict(img_array)
    max_prob = np.max(prediction)
    predicted_class = np.argmax(prediction)

    print(confidence_threshold)  # Add this line

    if max_prob >= confidence_threshold:
        predicted_label = class_labels[predicted_class]
    else:
        predicted_label = "unknown"

    filenames.append(image_file)
    predictions.append(predicted_label)  # Corrected line
    confidences.append(max_prob) #add confidence level.

# Create a DataFrame and save to CSV
results_df = pd.DataFrame({'filename': filenames, 'prediction': predictions, 'confidence': confidences}) #add confidence level.
results_df.to_csv('test_results.csv', index=False)

print("Evaluation complete. Results saved to test_results.csv")