"""
Image Classification Training Script (Transfer Learning)
Trains a CNN model on cats vs dogs dataset using ImageNet pretrained weights
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
DATA_DIR = 'data/archive/train'
MODEL_SAVE_DIR = 'examples'
IMG_HEIGHT = 224  # Standard ImageNet input size
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
BASE_MODEL_NAME = 'MobileNetV2'  # Options: VGG16, ResNet50, MobileNetV2, EfficientNetB0

# Create model save directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def create_transfer_learning_model(base_model_name='MobileNetV2', input_shape=(224, 224, 3), num_classes=2, trainable_layers=0):
    """
    Create a transfer learning model using ImageNet pretrained weights

    Args:
        base_model_name: Name of the pretrained model to use
        input_shape: Input image shape
        num_classes: Number of output classes
        trainable_layers: Number of layers from the end to make trainable (0 = freeze all base layers)

    Available base models:
    - VGG16: Simple and effective, larger model
    - ResNet50: Residual connections, good performance
    - MobileNetV2: Lightweight, fast inference
    - EfficientNetB0: Balanced efficiency and accuracy
    """

    # Load pretrained base model
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")

    # Freeze base model layers initially
    base_model.trainable = False

    # If trainable_layers > 0, unfreeze the last N layers
    if trainable_layers > 0:
        base_model.trainable = True
        # Freeze all layers except the last trainable_layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model

def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def main():
    print("=" * 60)
    print("Image Classification Training Script (Transfer Learning)")
    print(f"Using {BASE_MODEL_NAME} pretrained on ImageNet")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )

    print(f"\nLoading training data from: {DATA_DIR}")
    print(f"Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Batch size: {BATCH_SIZE}")

    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    # Create validation generator
    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    print(f"\nNumber of training samples: {train_generator.samples}")
    print(f"Number of validation samples: {validation_generator.samples}")
    print(f"Number of classes: {train_generator.num_classes}")
    print(f"Class indices: {train_generator.class_indices}")

    # Create model with transfer learning
    print(f"\nCreating transfer learning model using {BASE_MODEL_NAME}...")
    model, base_model = create_transfer_learning_model(
        base_model_name=BASE_MODEL_NAME,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=train_generator.num_classes,
        trainable_layers=0  # Start with all base layers frozen
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Calculate total and trainable parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Base model ({BASE_MODEL_NAME}) is frozen: {not base_model.trainable}")

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_SAVE_DIR, f'cats_dogs_{BASE_MODEL_NAME.lower()}_transfer_{timestamp}.h5')

    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 60)

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_DIR, f'cats_dogs_{BASE_MODEL_NAME.lower()}_transfer_final.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Plot training history
    plot_path = os.path.join(MODEL_SAVE_DIR, f'training_history_{BASE_MODEL_NAME.lower()}_transfer_{timestamp}.png')
    plot_training_history(history, plot_path)

    # Evaluate model on validation set
    print("\n" + "=" * 60)
    print("Evaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Print best model info
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Base model: {BASE_MODEL_NAME} (pretrained on ImageNet)")
    print(f"Best model saved to: {model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history plot: {plot_path}")
    print("=" * 60)

    # Optional: Fine-tuning phase
    print("\n" + "=" * 60)
    print("TIP: For fine-tuning, you can unfreeze some base layers and train again")
    print("Set trainable_layers > 0 in create_transfer_learning_model()")
    print("Use a lower learning rate (e.g., 1e-5) for fine-tuning")
    print("=" * 60)

    return model, history

if __name__ == "__main__":
    model, history = main()