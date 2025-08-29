# Install the required libraries first!
!pip install tensorflowjs

import tensorflow as tf
import tensorflowjs as tfjs
import os
import zipfile

print("TensorFlow version:", tf.__version__)

# --- 1. Mount Google Drive ---
# This will prompt you to authorize Colab to access your Google Drive.
from google.colab import drive
drive.mount('/content/drive')
print("Google Drive mounted successfully.")

# --- 2. Unzip Model and Data ---
DRIVE_PATH = '/content/drive/My Drive/'
MODEL_ZIP_PATH = os.path.join(DRIVE_PATH, 'tfjs_food_model.zip')
DATA_ZIP_PATH = os.path.join(DRIVE_PATH, 'nigerian_food_dataset.zip')

BASE_MODEL_PATH = '/content/base_food_model'
NIGERIAN_DATA_PATH = '/content/nigerian_food_dataset'

print("Unzipping base model...")
with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(BASE_MODEL_PATH)

print("Unzipping Nigerian food dataset...")
with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall('/') # Extract to root to get the nigerian_food_dataset folder

print("Files unzipped.")

# --- 3. Load the Base Food-101 Model ---
# Note: We need to load the original Keras model, not the TFJS one.
# The previous script should be modified to save the keras model.
# Let's assume for now the user has the keras model saved.
# This is a point of failure, I will address this in the message.
# For now, let's write the code assuming the Keras model exists.
# I will have to ask the user to re-run the first script with a modification.

# Let's pivot. It's too complex to ask the user to re-run the first script.
# I will rebuild the first model from scratch within THIS script.
# It's less efficient for the user, but more robust and less prone to error.
# The dataset is cached, so it will be fast.

# --- NEW PLAN ---

# 1. Load Food-101 from TFDS (fast from cache)
print("Loading Food-101 dataset from TFDS (from cache)...")
(train_data, validation_data), ds_info = tfds.load(
    'food101',
    split=['train', 'validation'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
FOOD101_CLASSES = ds_info.features['label'].names
print("Food-101 dataset loaded.")

# 2. Load Nigerian Food Dataset
print("Loading Nigerian food dataset...")
IMG_SIZE = 224
BATCH_SIZE = 32

nigerian_dataset = tf.keras.utils.image_dataset_from_directory(
    NIGERIAN_DATA_PATH,
    labels='inferred',
    label_mode='int',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
NIGERIAN_CLASSES = nigerian_dataset.class_names
print(f"Found {len(NIGERIAN_CLASSES)} Nigerian food classes: {NIGERIAN_CLASSES}")

# 3. Combine Labels
FINAL_CLASSES = FOOD101_CLASSES + NIGERIAN_CLASSES
NUM_FINAL_CLASSES = len(FINAL_CLASSES)
print(f"Total classes will be {NUM_FINAL_CLASSES}.")

# 4. Preprocess both datasets
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

nigerian_dataset = nigerian_dataset.map(preprocess_image).prefetch(tf.data.AUTOTUNE)

# We don't need to re-preprocess the TFDS one if we build the model from scratch.

# 5. Build the base model and fine-tune on Food-101
print("Building and training base Food-101 model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(FOOD101_CLASSES), activation='softmax')(x)
food101_model = tf.keras.Model(inputs, outputs)
food101_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
food101_model.fit(train_data.map(preprocess_image).batch(BATCH_SIZE), epochs=5, validation_data=validation_data.map(preprocess_image).batch(BATCH_SIZE))
print("Base model training complete.")

# 6. Create a new model for final fine-tuning
# Let's unfreeze some layers of the base model to adapt it better
base_model.trainable = True
# Let's unfreeze the top 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# We need to rebuild the model with the new number of classes
new_outputs = tf.keras.layers.Dense(NUM_FINAL_CLASSES, activation='softmax')(food101_model.layers[-2].output) # Get output of the dropout layer
# This is getting too complex. Let's simplify the strategy.

# --- SIMPLER, MORE ROBUST STRATEGY ---
# Let's not train two models. Let's combine the datasets and train ONE model.
# This is much cleaner and more effective.

# (Re-writing the script from scratch with the new strategy)
# The user already has the nigerian dataset zipped in their drive.

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflowjs as tfjs
import os
import zipfile
from google.colab import drive

print("TensorFlow version:", tf.__version__)
drive.mount('/content/drive')

# --- 1. Unzip Nigerian Food Data ---
DRIVE_PATH = '/content/drive/My Drive/'
DATA_ZIP_PATH = os.path.join(DRIVE_PATH, 'nigerian_food_dataset.zip')
NIGERIAN_DATA_PATH = '/content/nigerian_food_dataset'
with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall('/')
print("Nigerian food dataset unzipped.")

# --- 2. Load Both Datasets ---
IMG_SIZE = 224
BATCH_SIZE = 32

# Load Nigerian dataset from the directory
nigerian_ds = tf.keras.utils.image_dataset_from_directory(
    NIGERIAN_DATA_PATH,
    label_mode='int',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)
NIGERIAN_CLASSES = nigerian_ds.class_names
print(f"Found {len(NIGERIAN_CLASSES)} Nigerian food classes.")

# Load Food-101 dataset from TFDS
(food101_train, food101_val), ds_info = tfds.load(
    'food101',
    split=['train', 'validation'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
FOOD101_CLASSES = ds_info.features['label'].names
print(f"Loaded {len(FOOD101_CLASSES)} Food-101 classes.")

# --- 3. Combine Labels and Create a Label Mapping ---
# We need to remap the labels of the Food-101 dataset so they don't conflict
# with the Nigerian dataset labels (which start from 0).
FINAL_LABELS = NIGERIAN_CLASSES + FOOD101_CLASSES
NUM_FINAL_CLASSES = len(FINAL_LABELS)

food101_label_offset = len(NIGERIAN_CLASSES)
def remap_food101_labels(image, label):
    return image, label + food101_label_offset

food101_train = food101_train.map(remap_food101_labels)
food101_val = food101_val.map(remap_food101_labels)
print("Labels remapped.")

# --- 4. Combine Datasets ---
# To make training fair, we should ideally balance the datasets.
# Food-101 has 750 images per class. The Nigerian one has ~20.
# Let's take a smaller sample from Food-101 to balance it.
# This also makes training much faster.
SAMPLES_PER_CLASS = 50 # Let's take 50 images from each Food-101 class
food101_train = food101_train.filter(lambda image, label: tf.random.uniform(()) < (SAMPLES_PER_CLASS / 750)).shuffle(1000)

# Now combine them
combined_train_ds = nigerian_ds.concatenate(food101_train)
# For validation, let's just use the Food-101 validation set for simplicity
combined_val_ds = food101_val

print("Datasets combined.")

# --- 5. Preprocess the Combined Dataset ---
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

combined_train_ds = combined_train_ds.map(preprocess_image).shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
combined_val_ds = combined_val_ds.map(preprocess_image).prefetch(tf.data.AUTOTUNE)

# --- 6. Build and Train the Final Model ---
print("Building the final model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Start with the base frozen

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_FINAL_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("--- Initial Training (Head Only) ---")
model.fit(combined_train_ds, epochs=5, validation_data=combined_val_ds)

# --- 7. Fine-Tuning ---
base_model.trainable = True
for layer in base_model.layers[:-40]: # Unfreeze top 40 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Use a low learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("\n--- Fine-Tuning (Unfrozen Layers) ---")
model.fit(combined_train_ds, epochs=5, validation_data=combined_val_ds)
print("Model training complete.")

# --- 8. Save Final Model and Labels ---
with open('final_labels.txt', 'w') as f:
    f.write('\n'.join(FINAL_LABELS))
print("Final labels saved to 'final_labels.txt'")

tfjs_model_path = 'final_tfjs_model'
tfjs.converters.save_keras_model(model, tfjs_model_path)
print(f"Final model saved to '{tfjs_model_path}'")
print("\n--- Process Complete ---")
print("Please download 'final_labels.txt' and the 'final_tfjs_model' directory (after zipping it).")
