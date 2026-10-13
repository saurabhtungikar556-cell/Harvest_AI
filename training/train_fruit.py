import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Dynamically resolve repository root relative to the training/ directory
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(TRAINING_DIR)

TRAIN_DIR = os.path.join(BASE_DIR, "dataset_fruit", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset_fruit", "test")
# Unified naming convention to prevent pipeline fractures
SAVE_PATH = os.path.join(BASE_DIR, "models", "fruit_model.h5")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

def enforce_determinism(seed=42):
    """Locks TensorFlow RNG states for academic reproducibility."""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

def main():
    enforce_determinism()

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        zoom_range=0.2, 
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    test_data = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )

    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])

    # Implement callbacks to prevent overfitting and capture optimal weights
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    callbacks = [
        ModelCheckpoint(filepath=SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    ]

    print("Starting deterministic training pipeline with checkpointing...")
    model.fit(train_data, validation_data=test_data, epochs=EPOCHS, callbacks=callbacks)
    print(f"Optimal model architecture and weights secured at {SAVE_PATH}")

if __name__ == "__main__":
    main()