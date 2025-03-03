import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

# Path to your image dataset
data_dir = 'images'

# Image dimensions
img_height, img_width = 100, 100
batch_size = 8

# Custom data augmentation function to apply hue adjustments using tf.image.random_hue
def augment_image(image, label):
    image = tf.image.random_brightness(image, max_delta=0.2)  # Adjust brightness
    # image = tf.image.random_contrast(image, lower=0.8, upper=1.2) # Adjust contrast
    # image = tf.image.random_saturation(image, lower=0.8, upper=1.2) # Adjust saturation
    # image = tf.image.random_hue(image, max_delta=0.2) # Adjust hue
    image = tf.image.random_flip_left_right(image) # Random horizontal flips
    image = tf.image.random_flip_up_down(image) # Random vertical flips/
    # image = tf.image.random_crop(image, size=[img_height, img_width, 3]) # Random crop to target size
    return image, label

# ImageDataGenerator for basic preprocessing and augmentation (excluding hue)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2
)

# ImageDataGenerator for validation data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

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

# Convert generators to tf.data.Dataset for custom augmentation
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32)))

# Apply custom augmentation function to the training dataset
train_dataset = train_dataset.map(augment_image)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, validation_generator.num_classes), dtype=tf.float32)))

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.7),  # Dropout layer added here
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.7),  # Dropout layer added here
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
# ])

# Compile the model
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
# loss='categorical_crossentropy',
# metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# Train the model
epochs = 200
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save('image_classifier.keras')

print("Training complete. Model saved as image_classifier.keras")
# --- Confusion Matrix Code ---

predictions = model.predict(validation_dataset, steps=validation_generator.samples // batch_size)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = validation_generator.classes[validation_generator.index_array[:validation_generator.samples // batch_size * batch_size]]

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()