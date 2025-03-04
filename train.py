import tensorflow as tf
from tensorflow.keras.applications import ResNet50  # Import ResNet50 pre-trained model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  # Import necessary layers
from tensorflow.keras.models import Model  # Import the Model class for creating functional models
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator for data augmentation
import numpy as np
from sklearn.metrics import confusion_matrix  # Import confusion matrix for evaluation
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your image dataset
data_dir = 'images'

# Image dimensions
img_height, img_width = 100, 100
batch_size = 8

# Custom data augmentation function
def augment_image(image, label):
    image = tf.image.random_brightness(image, max_delta=0.2)  # Adjust brightness
    image = tf.image.random_flip_left_right(image)  # Random horizontal flips
    image = tf.image.rgb_to_grayscale(image) #Converts to grayscale
    image = tf.image.grayscale_to_rgb(image) #Converts back to rgb, so that the pretrained model can use it.
    return image, label

# ImageDataGenerator for basic preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to [0, 1]
    rotation_range=20,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Random shear transformations
    zoom_range=0.2,  # Random zooms
    validation_split=0.2  # Split data for validation

)

# ImageDataGenerator for validation data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create data generators from directory
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Categorical labels (one-hot encoded)
    subset='training'  # Use training subset
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

# Load ResNet50 pre-trained model (excluding top classification layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce spatial dimensions
x = Dense(128, activation='relu')(x)  # Dense layer with ReLU activation
x = Dropout(0.5)(x)  # Dropout for regularization
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Output layer with softmax activation

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tuning
base_model.trainable = True
fine_tune_at = 150  # Unfreeze the last 100 layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
              loss='categorical_crossentropy',  # Categorical crossentropy loss
              metrics=['accuracy'])  # Track accuracy


# Train the model
epochs = 200
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Early stopping callback


history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

history_fine = model.fit(
    train_dataset,
    epochs=100 + 25,  # Train for more epochs
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('resnet50_image_classifier.keras')

print("Training complete. Model saved as resnet50_image_classifier.keras")

# --- Confusion Matrix Code ---

# Generate predictions on the validation dataset
predictions = model.predict(validation_dataset, steps=validation_generator.samples // batch_size)
predicted_classes = np.argmax(predictions, axis=1)  # Get predicted class indices

# Get true class indices from the validation generator
true_classes = validation_generator.classes[validation_generator.index_array[:validation_generator.samples // batch_size * batch_size]]

# Create confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()