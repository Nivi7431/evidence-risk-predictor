import os
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Define image dimensions
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# Paths for training and validation data
TRAIN_DIR = 'data/train'
VALIDATION_DIR = 'data/validation'

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'  # Use 'categorical' for multiple classes
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Check if data generators are correctly detecting 3 classes
assert train_generator.num_classes == 3, "Train generator is detecting more/less than 3 classes"
assert validation_generator.num_classes == 3, "Validation generator is detecting more/less than 3 classes"

# CNN Model Definition with 3 output classes
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: blood, cloth, fingerprint
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Save the model to the 'models/' directory
model.save('models/evidence_type_cnn.h5')
