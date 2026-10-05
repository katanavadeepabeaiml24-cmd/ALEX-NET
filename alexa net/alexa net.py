# ============================================================
# SIMPLE ALEXNET IMAGE CLASSIFICATION
# CIFAR-10 DATASET
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

print("TensorFlow:", tf.__version__)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Train:", x_train.shape)
print("Test :", x_test.shape)

# Class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# ------------------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------------------

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to 1D
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

print("Preprocessing completed!")

# ------------------------------------------------------------
# 3. ALEXNET MODEL
# ------------------------------------------------------------

model = models.Sequential([

    # Input
    layers.Input(shape=(32, 32, 3)),

    # Conv Layer 1
    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Conv Layer 2
    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Conv Layer 3
    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    # Conv Layer 4
    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    # Pooling
    layers.MaxPooling2D((2, 2)),

    # Fully Connected
    layers.Flatten(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.5),

    # Output
    layers.Dense(
        10,
        activation="softmax"
    )
])

# ------------------------------------------------------------
# 4. SHOW MODEL
# ------------------------------------------------------------

model.summary()

# ------------------------------------------------------------
# 5. COMPILE
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")

# ------------------------------------------------------------
# 6. TRAIN MODEL
# ------------------------------------------------------------

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    verbose=1
)

# ------------------------------------------------------------
# 7. TEST MODEL
# ------------------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print("Test Accuracy:",
      round(test_accuracy * 100, 2), "%")

print("Test Loss:",
      round(test_loss, 4))

# ------------------------------------------------------------
# 8. ACCURACY GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("AlexNet Training and Validation Accuracy")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 9. LOSS GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AlexNet Training and Validation Loss")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 10. PREDICTIONS
# ------------------------------------------------------------

predictions = model.predict(
    x_test,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

# ------------------------------------------------------------
# 11. DISPLAY SAMPLE PREDICTIONS
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

for i in range(12):

    plt.subplot(3, 4, i + 1)

    plt.imshow(x_test[i])

    predicted = class_names[predicted_classes[i]]
    actual = class_names[y_test[i]]

    confidence = np.max(predictions[i]) * 100

    plt.title(
        "Pred: " + predicted +
        "\nActual: " + actual +
        "\nConfidence: " + str(round(confidence, 1)) + "%"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. SINGLE IMAGE PREDICTION
# ------------------------------------------------------------

index = 0

image = x_test[index]

prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)

predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("\n==============================")
print("SINGLE IMAGE PREDICTION")
print("==============================")

print("Actual    :", class_names[y_test[index]])
print("Predicted :", class_names[predicted_class])
print("Confidence:", round(confidence, 2), "%")

plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.title(
    "Predicted: " +
    class_names[predicted_class] +
    "\nConfidence: " +
    str(round(confidence, 2)) + "%"
)
plt.axis("off")
plt.show()

# ------------------------------------------------------------
# 13. SAVE MODEL
# ------------------------------------------------------------

model.save("alexnet_model.keras")

print("\nModel saved successfully!")
print("File: alexnet_model.keras")

print("\n==============================")
print("PROJECT COMPLETED")
print("==============================")