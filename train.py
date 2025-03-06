import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import albumentations as A  # Import Albumentations

os.environ['QT_QPA_PLATFORM_NAME'] = 'windows'

# Path to your image dataset
data_dir = 'images'

# Image dimensions
img_height, img_width = 100, 100
batch_size = 8

# Albumentations augmentations
transform = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),
    # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5)
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-20, 20), shear=(-5, 5), p=0.5), #replace shiftscalerotate
    A.GaussNoise(p=0.5) #replace and simplify gaussnoise
])

def augment_image(image):
    augmented = transform(image=image)
    return augmented['image']

# ImageDataGenerator for basic preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
)

# ImageDataGenerator for validation data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

# Create data generators from directory
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Place the print statement for debugging
for images, labels in train_generator:
    print(f"train_generator image shape: {images.shape}, label shape: {labels.shape}")
    break

# Keras Generator Test (Place this here)-for debugging
for images, labels in train_generator:
    print(f"Keras Generator image shape: {images.shape}")
    print(f"Keras Generator label shape: {labels.shape}")
    break  # Print only the first batch

validation_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Insert print statements here:
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Batch size: {batch_size}")
print(f"Steps per epoch: {train_generator.samples // batch_size}")
print(f"Validation steps: {validation_generator.samples // batch_size}")

def augment_image(image):
    # Assuming your image augmentation logic is here
    # ... your augmentation operations ...

    # Example: Placeholder augmentation (replace with your actual augmentation)
    augmented_image = image  # Replace with your actual augmentation result

    # Correcting the type: Ensure the output is uint8
    augmented_image = augmented_image.astype(np.uint8)  # Explicitly cast to uint8

    return augmented_image

# Function to load and augment images
def load_and_augment(image, label):
    print(f"Original image shape: {image.shape}")

    def augment_wrapper(image):
        return augment_image(image)

    augmented_image = tf.numpy_function(func=augment_wrapper, inp=[image], Tout=tf.uint8)
    augmented_image = tf.cast(augmented_image, tf.float32)

    print(f"Shape after cast: {augmented_image.shape}")

    augmented_image = augmented_image / 255.0  # Normalize augmented_image

    print(f"Before set_shape: augmented_image.shape={augmented_image.shape}, label.shape={label.shape}")

    # Explicitly set the shape here:
    augmented_image.set_shape((img_height, img_width, 3))

    # label.set_shape((train_generator.num_classes,))
    # label.set_shape((train_generator.num_classes,))
    # label.set_shape((1, train_generator.num_classes)) # set to 2D tensor
    label = tf.convert_to_tensor(label, dtype=tf.float32)

    print(f"After set_shape: augmented_image.shape={augmented_image.shape}, label.shape={label.shape}")

    return augmented_image, label
# def load_and_augment(image, label):
#     # image = cv2.imread(image_path.numpy().decode('utf-8'))
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # image = cv2.resize(image, (img_height, img_width))
#     print(f"Original image shape: {image.shape}") #added print statement
#     def augment_wrapper(image):
#         return augment_image(image)

#     augmented_image = tf.numpy_function(func=augment_wrapper, inp=[image], Tout=tf.uint8) #Tout needs to be uint8, since augment_image returns numpy uint8 array.
#     augmented_image = tf.cast(augmented_image, tf.float32) # Cast to float32 after numpy_function.
#     augmented_image = tf.squeeze(augmented_image) # Add squeeze
#     print(f"Shape after cast: {augmented_image.shape}") #Added print statement.
#     # image = augment_image(image)
#     image = image / 255.0

#     print(f"Before set_shape: image.shape={image.shape}, label.shape={label.shape}")

#     # Set the shape to (None, 100, 100, 3) to match the original shape
#     augmented_image.set_shape((None, img_height, img_width, 3))

#     # Explicitly set the tensor shapes
#     image.set_shape((img_height, img_width, 3))
#     label.set_shape((train_generator.num_classes,)) # Make sure this matches your label size.
#     image = tf.convert_to_tensor(image, dtype=tf.float32)
#     label = tf.convert_to_tensor(label, dtype=tf.float32)

#     print(f"After set_shape: image.shape={image.shape}, label.shape={label.shape}")

#     return image, label

def wrapped_train_generator():
    for images, labels in train_generator:  # Use the correct instance
        for i in range(images.shape[0]):  # Iterate through the batch
            print(f"Yielding image shape: {images[i].shape}, label shape: {labels[i].shape}")
            yield images[i], labels[i]
        # print(f"Generator image shape: {images.shape}")
        # print(f"Generator label shape: {labels.shape}")
        # yield images, labels

def simple_generator():
    for i in range(5):  # Generate 5 simple batches
        images = np.ones((32, 100, 100, 3), dtype=np.float32) * i  # Simple images
        labels = np.ones((32, 2), dtype=np.float32) * i  # Simple labels
        print(f"Simple Generator image shape: {images.shape}")
        print(f"Simple Generator label shape: {labels.shape}")
        yield images, labels

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: wrapped_train_generator(),
    output_signature=(tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32), 
    tf.TensorSpec(shape=(train_generator.num_classes), dtype=tf.float32))
)

#for debugging
for images, labels in train_dataset.take(1):
    print(f"Dataset image shape: {images.shape}")
    print(f"Dataset label shape: {labels.shape}")

def set_shapes(image, label):
    image.set_shape((img_height, img_width, 3))
    label.set_shape((train_generator.num_classes,))
    return image, label
# Map the augmentation function
# train_dataset = train_dataset.map(lambda image, label: tf.numpy_function(
#     load_and_augment, 
#     inp=[image, label], 
#     Tout=[tf.float32, tf.float32]), 
#     num_parallel_calls=tf.data.AUTOTUNE)

# train_dataset = train_dataset.map(
#     lambda image, label:  tf.py_function(
#         load_and_augment,
#         inp=[image, label],
#         Tout=[tf.float32, tf.float32],
#         # output_signature=(
#         #     tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32),
#         #     tf.TensorSpec(shape=(train_generator.num_classes,), dtype=tf.float32)
#         # )
#     ),
#     num_parallel_calls=tf.data.AUTOTUNE
# ).map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size) # Add .batch(batch_size) here!

train_dataset = train_dataset.map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

# validation_dataset = tf.data.Dataset.from_generator(
#     lambda: validation_generator,
#     output_signature=(tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, validation_generator.num_classes), dtype=tf.float32))).batch(batch_size) #add batch size


def wrapped_validation_generator():
    for images, labels in validation_generator:
        for i in range(images.shape[0]):
            yield images[i], labels[i]

validation_dataset = tf.data.Dataset.from_generator(
    # lambda: validation_generator,
    lambda: wrapped_validation_generator(),
    output_signature=(tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32), tf.TensorSpec(shape=(validation_generator.num_classes,), dtype=tf.float32))
).batch(batch_size)

# Load ResNet50 pre-trained model (excluding top classification layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tuning
base_model.trainable = True
fine_tune_at = 150
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 200
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#print the model for debugging
model.summary()

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
    epochs=100 + 25,
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

# Function to perform color check and prediction
def predict_with_color_check(model, generator, lime_hue_range, lime_saturation_range, threshold, confidence_threshold):
    predictions = []
    true_classes = []
    for image_path, label in tf.data.Dataset.from_tensor_slices((generator.filenames, generator.classes)):
        image_path_str = os.path.join(generator.directory, image_path.numpy().decode('utf-8')) #decode EagerTensor to string.
        image = cv2.imread(image_path_str)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # lower_lime = np.array([lime_hue_range[0], lime_saturation_range[0], 100])
        # upper_lime = np.array([lime_hue_range[1], lime_saturation_range[1], 255])
        lower_lime = np.array([lime_hue_range[0], lime_saturation_range[0], lime_value_range[0]])
        upper_lime = np.array([lime_hue_range[1], lime_saturation_range[1], lime_value_range[1]])

        mask = cv2.inRange(image_hsv, lower_lime, upper_lime)
        # Display the mask
        cv2.imshow('Mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        total_pixels = mask.shape[0] * mask.shape[1]
        non_lime_pixels = total_pixels - cv2.countNonZero(mask)
        deviation_percentage = (non_lime_pixels / total_pixels) * 100
        print(f"Image: {image_path_str}, Deviation Percentage: {deviation_percentage:.2f}%") #Added print statement.

        # print(f"Image: {image_path_str}") #added printing image name
        # Print Hue Values of Lime Pixels:
        # for y in range(image_hsv.shape[0]):
        #     for x in range(image_hsv.shape[1]):
        #         # if mask[y, x] == 255:  # If pixel is within the mask (lime color)
        #         hue = image_hsv[y, x, 0]
        #         saturation = image_hsv[y,x,1]
        #         value = image_hsv[y,x,2]
        #         print(f"  Hue: {hue}, Saturation: {saturation}, Value: {value}")
        
        if "D16-base.png" in image_path_str: #Conditional Check
            print("HSV values for D16-base.png:")
            for y in range(image_hsv.shape[0]):
                for x in range(image_hsv.shape[1]):
                    hue = image_hsv[y, x, 0]
                    saturation = image_hsv[y, x, 1]
                    value = image_hsv[y, x, 2]
                    print(f"  Pixel ({x}, {y}): Hue={hue}, Saturation={saturation}, Value={value}")

        if deviation_percentage > threshold:
            predictions.append(np.array([0.0, 0.0, 1.0])) # unknown class.
        else:
            img = tf.keras.preprocessing.image.load_img(image_path_str, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = model.predict(img_array)
            max_prob = np.max(prediction)

            if max_prob < confidence_threshold:
                predictions.append(np.array([0.0, 0.0, 1.0])) # unknown class.
            else:
                predictions.append(prediction[0])
        true_classes.append(label.numpy())
    return np.array(predictions), np.array(true_classes)

# Define color check parameters,from manully picked lime color from sample images
# lime_hue_range = (85, 91)
# lime_saturation_range = (230, 260)
lime_hue_range = (86, 95)
lime_saturation_range = (200, 255)
lime_value_range = (170, 255)
threshold = 50 #percentage of non-lime pixels.
confidence_threshold = 0.8

# Make predictions on the validation dataset with color check
predictions, true_classes = predict_with_color_check(model, validation_generator, lime_hue_range, lime_saturation_range, threshold, confidence_threshold)

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