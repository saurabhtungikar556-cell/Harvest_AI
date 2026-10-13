import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from config import IMG_SIZE, FRESH_MODEL_PATH

# Dynamically resolve repository root relative to this script's location
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
TRAIN_DIR = os.path.join(BASE_DIR, "dataset_freshness", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset_freshness", "test")

BATCH_SIZE = 32
EPOCHS = 25

def enforce_determinism(seed=42):
    """Locks TensorFlow RNG states for academic reproducibility."""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

def get_dataset_files(root_path):
    filenames = []
    categories = []
    classes = ['fresh', 'rotten']
    
    for category in classes:
        class_folder = os.path.join(root_path, category)
        if not os.path.exists(class_folder):
            print(f"Folder missing: {class_folder}")
            continue
            
        for root, _, files in os.walk(class_folder):
            # Bypass for local dataset pollution
            if category == 'fresh' and 'rotten' in root:
                continue

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filenames.append(os.path.join(root, file))
                    categories.append(category)

    df = pd.DataFrame({'filename': filenames, 'class': categories})
    print(f"Loaded {len(df)} images from {root_path}")
    return df

def main():
    enforce_determinism()
    
    df_train = get_dataset_files(TRAIN_DIR)
    df_test = get_dataset_files(TEST_DIR)

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3]
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_dataframe(
        df_train, x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )

    test_data = test_gen.flow_from_dataframe(
        df_test, x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )

    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Starting deterministic training pipeline...")
    
    model.fit(train_data, validation_data=test_data, epochs=EPOCHS)

    os.makedirs(os.path.dirname(FRESH_MODEL_PATH), exist_ok=True)
    model.save(FRESH_MODEL_PATH)
    print(f"Model saved to {FRESH_MODEL_PATH}")

if __name__ == "__main__":
    main()