# testing code through web cam 

import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("symbol_classifier_v2.h5")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Press 'q' to quit manually")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

   
    symbol_number = predicted_index + 1


    if predicted_index < 16:
        label = "REAL"
        color = (0, 255, 0)
    else:
        label = "FAKE"
        color = (0, 0, 255)

    display_text = f"Symbol {symbol_number} | {label} | {confidence*100:.2f}%"
    cv2.putText(frame, display_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Symbol Detection", frame)

    if confidence >= 0.70:
        print(f"Detected: Symbol {symbol_number} ({label}) with {confidence*100:.2f}% confidence")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
