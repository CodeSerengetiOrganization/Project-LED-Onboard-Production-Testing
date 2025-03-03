import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

# Path to your image dataset
data_dir = 'images'

# Image dimensions
img_height, img_width = 100, 100
batch_size = 8

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

# Function to load and augment images, including color deviation calculation
def load_and_augment(image_path, label):
    image = cv2.imread(image_path.numpy().decode('utf-8'))
    if image is None: #check if the image is loaded.
        print(f"Error loading image: {image_path.numpy().decode('utf-8')}")
        return np.zeros((img_height,img_width,3),dtype=np.float32), 0.0, label #return default values.
    deviation = calculate_color_deviation(image_path.numpy().decode('utf-8'))
    image = cv2.resize(image, (img_height, img_width))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0

    image = image.astype(np.float32) #convert to float32
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)

    return image, deviation, label

# TensorFlow wrapper for load_and_augment function
def tf_load_and_augment(image_path, label):
    image, deviation, label = tf.py_function(load_and_augment, inp=[image_path, label], Tout=[tf.float32, tf.float32, tf.int32])
    image.set_shape([img_height, img_width, 3])
    deviation.set_shape([])
    label.set_shape([])
    label = tf.one_hot(label, depth=train_generator.num_classes)
    label.set_shape([train_generator.num_classes])
    return (image, deviation), label

# Function to create TensorFlow datasets from generators
def create_dataset(generator):
    image_paths = [os.path.join(generator.directory, filename) for filename in generator.filenames]
    labels = generator.classes

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(tf_load_and_augment)
    dataset = dataset.batch(batch_size)
    return dataset

# ImageDataGenerator for basic preprocessing and augmentation
train_datagen = ImageDataGenerator(validation_split=0.2)

# ImageDataGenerator for validation data (only rescaling)
validation_datagen = ImageDataGenerator(validation_split=0.2)

# Create data generators from directory
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create TensorFlow datasets
train_dataset = create_dataset(train_generator)
validation_dataset = create_dataset(validation_generator)

# Define input layers for ResNet50 and color deviation
image_input = Input(shape=(img_height, img_width, 3))
deviation_input = Input(shape=(1,))

# Load ResNet50 pre-trained model (excluding top classification layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)

# Freeze the base model layers
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Concatenate()([x, deviation_input])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=[image_input, deviation_input], outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 200
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

# Fine-tuning
base_model.trainable = True
fine_tune_at = 150
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_dataset,
    epochs=100 + 25,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('resnet50_image_classifier.keras')
print("Training complete. Model saved as resnet50_image_classifier.keras")

# --- Confusion Matrix Code ---

# Function to make predictions with color deviation
def predict_with_deviation(model, generator):
    predictions = []
    true_classes = []
    for image_path, label in tf.data.Dataset.from_tensor_slices((generator.filenames, generator.classes)):
        image, deviation, _ = load_and_augment(image_path, label)
        prediction = model.predict((np.expand_dims(image, axis=0), np.expand_dims(deviation, axis=0)))
        predictions.append(prediction[0])
        true_classes.append(label)
    return np.array(predictions), np.array(true_classes)

# Make predictions on the validation dataset
predictions, true_classes = predict_with_deviation(model, validation_generator)

# Get predicted class indices
predicted_classes = np.argmax(predictions, axis=1)

# Create confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()