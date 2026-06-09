import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import pandas as pd


img_height = 224
img_width = 224
batch_size = 16
test_dir = 'DATASET/test'

#LOAD MODEL
model = tf.keras.models.load_model("raf_resnet50_final.keras")
print("Model loaded successfully!")

#TEST GENERATOR
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False   # КРИТИЧНО
)

#PREDICTIONS
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = list(test_generator.class_indices.keys())

#CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

#CLASSIFICATION REPORT
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)


df = pd.DataFrame(report).transpose()

df['recall'][:-3].plot(kind='bar')
plt.title("Recall per class")
plt.ylabel("Recall")
plt.ylim(0,1)
plt.show()
