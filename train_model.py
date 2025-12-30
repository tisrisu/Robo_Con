import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense , GlobalAveragePooling2D , Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.regularizers import l2





base_dir = "../dataset/processed"
train_dir = os.path.join(base_dir , "train")
test_dir = os.path.join (base_dir , "test")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

train_gen = ImageDataGenerator( rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
    )

test_gen = ImageDataGenerator(rescale = 1./255)

train_data = train_gen.flow_from_directory(

    train_dir ,
    target_size  = (IMG_SIZE,IMG_SIZE) ,
    batch_size =BATCH_SIZE ,
    class_mode = "categorical" ,
    shuffle = True

)

test_data = test_gen.flow_from_directory (

    test_dir ,
    target_size = (IMG_SIZE , IMG_SIZE) ,
    batch_size = BATCH_SIZE ,
    shuffle = False

)

# Get class indices
labels = train_data.classes

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

# Convert to dictionary (required by Keras)
class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)


print("Classes detected:", train_data.class_indices)

base_model = MobileNetV2 (

    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)

)

base_model.trainable=False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense (32 , activation = "relu" , kernel_regularizer=l2(0.001))(x)
x = Dropout (0.5)(x)
output = Dense (train_data.num_classes , activation = "softmax")(x)

model = Model(inputs = base_model.input , outputs = output)

model.compile (

    optimizer = Adam(learning_rate=5e-5) ,
    loss = "categorical_crossentropy" , 
    metrics = ["accuracy"]

)

early_stop = EarlyStopping(

    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20,
    callbacks=[early_stop],
    class_weight=class_weights
)

model.save("symbol_classifier.h5")

print(" Training complete. Model saved as symbol_classifier.h5")
print(train_data.class_indices)

for i in range(10):
    x, y = next(train_data)
    print("Label index:", np.argmax(y[0]))
