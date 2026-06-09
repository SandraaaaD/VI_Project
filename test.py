import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

import matplotlib.pyplot as plt

img_height = 224
img_width = 224
model_path = "raf_resnet50_final.keras"

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']



model = tf.keras.models.load_model(model_path)
print("Model loaded!")



image_path = 'slika_tazhna.jpg'   # <<< смени тука

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (img_width, img_height))

img = preprocess_input(img)



img = np.expand_dims(img, axis=0)

pred = model.predict(img)
pred_class = np.argmax(pred)
confidence = np.max(pred)

print("\nPredicted emotion:", class_names[pred_class])
print("Confidence:", round(confidence * 100, 2), "%")


pred = model.predict(img)
class_index = np.argmax(pred)
print(class_names[class_index])


pred = model.predict(img)
print(pred)

