import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

BASE_DIR = "../dataset/processed"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")



def rgb_to_gray(x):
    return tf.image.rgb_to_grayscale(x)

train_gen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: tf.image.rgb_to_grayscale(x),
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    shear_range=0.15,
    brightness_range=[0.3, 1.7]
)

test_gen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: tf.image.rgb_to_grayscale(x)
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    shuffle=True
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=16,
    shuffle=False
)

model = load_model("symbol_classifier_v2.h5")

for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=6,
    callbacks=[early_stop]
)

model.save("symbol_classifier_finetuned.h5")

print(" Fine-tuning complete. Model saved as symbol_classifier_finetuned.h5")
