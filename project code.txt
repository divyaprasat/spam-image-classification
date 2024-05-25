import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

X = tf.keras.Input(shape=[240, 240, 3], dtype=tf.float32, name="X")

resnet_base = ResNet50(weights='imagenet', include_top=False, input_tensor=X)

for layer in resnet_base.layers:
    layer.trainable = False

global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(resnet_base.output)

caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
caps1_n_dims = 8

dense1 = tf.keras.layers.Dense(caps1_n_maps * caps1_n_dims, activation=tf.nn.relu)(global_avg_pooling)

caps1_raw = tf.reshape(dense1, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name or "squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

caps1_output = squash(caps1_raw, name="caps1_output")

model = models.Model(inputs=X, outputs=caps1_output, name="CapsuleNetwork")

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])



import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_and_labels(folder_path, label, target_size=(240, 240)):
    images = []
    labels = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)
                img = img.convert('RGB')  
                # Resize the image
                img_resized = img.resize(target_size)
                img_array = np.array(img_resized)
                images.append(img_array)
                labels.append(label)
    except Exception as e:
        print(f"Error loading images from {folder_path}: {e}")

    return images, labels

spam_folder = "/Users/jai/Downloads/dataset/spam_resized"
ham_folder = "/Users/jai/Downloads/dataset/ham_resized"

spam_images, spam_labels = load_images_and_labels(spam_folder, label=1,target_size=(240, 240))  # Assign label 1 to spam
ham_images, ham_labels = load_images_and_labels(ham_folder, label=0,target_size=(240, 240))    # Assign label 0 to ham


image_shapes = set(img.shape for img in spam_images + ham_images)
if len(image_shapes) > 1:
    print("Images have different shapes. Resize or preprocess them to have a common size.")
else:
    all_images = np.concatenate([spam_images, ham_images], axis=0)
    all_labels = np.concatenate([spam_labels, ham_labels], axis=0)

 

   
train_images, temp_images, train_labels, temp_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)
print("Training set size:", len(train_images))
print("Validation set size:", len(val_images))
print("Testing set size:", len(test_images))


model = models.Sequential()
model.add(layers.Flatten(input_shape=(240,240,3))) #this line
model.add(layers.Dense(128, activation='relu', input_shape=(240, 240, 3)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if model.optimizer is not None:
    print("Model is compiled!")
else:
    print("Model is not compiled.")

history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
print("Predicted Labels:", predicted_labels)












from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

predicted_labels = model.predict(test_images)
predicted_labels = np.argmax(predicted_labels, axis=1)

accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average='weighted')
recall = recall_score(test_labels, predicted_labels, average='weighted')
f1 = f1_score(test_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

report = classification_report(test_labels, predicted_labels)
print("Classification Report:\n", report)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(240, 240)):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array[None, ...])
    return img_array, img_preprocessed

new_image_paths = ["/Users/jai/Downloads/dataset/ham_resized/zzz_0013_4950651bbd_m.jpg","/Users/jai/Downloads/dataset/spam_resized/erosiverefrigerate.jpg", "/Users/jai/Downloads/dataset/ham_resized/zzz_104_c1a35af827_m.jpg","/Users/jai/Downloads/dataset/spam_resized/cBA9i4p7Kf.jpg"]

for image_path in new_image_paths:
    # Load and preprocess the image
    original_image, new_image = load_and_preprocess_image(image_path)

    prediction = model.predict(new_image)

    predicted_class = 1 if prediction[0][0] > 0.5 else 0

    plt.imshow(original_image)
    plt.title(f"Predicted Class: {predicted_class}, Prediction Value: {prediction}")
    plt.show()
