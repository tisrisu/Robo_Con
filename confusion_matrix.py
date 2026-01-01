# to check the accuracy 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


model = load_model("symbol_classifier_final.h5")



test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    "../dataset/processed/test",
    target_size=(224, 224),
    batch_size=32,
    shuffle=False
)

y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes


cm = confusion_matrix(y_true, y_pred_classes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_data.class_indices.keys()
)

disp.plot(cmap="Blues", xticks_rotation=90)
plt.title("Confusion Matrix")
plt.show()

