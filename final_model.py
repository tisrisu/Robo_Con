import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = "../dataset/processed"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    shear_range=0.15,
    brightness_range=[0.4, 1.6]
)

test_gen = ImageDataGenerator(rescale=1./255)

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

model = load_model("symbol_classifier_finetuned.h5")

for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=3,
    callbacks=[early_stop]
)

model.save("symbol_classifier_final.h5")
print(" Fine-tuned model saved as symbol_classifier_final.h5")
