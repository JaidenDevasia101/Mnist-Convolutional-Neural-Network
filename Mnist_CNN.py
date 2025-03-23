import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize data to range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images[..., tf.newaxis]  # Add channel dimension
test_images = test_images[..., tf.newaxis]    # Add channel dimension

# Split the training data into training and validation sets
validation_images = train_images[-12000:]
validation_labels = train_labels[-12000:]
train_images = train_images[:-12000]
train_labels = train_labels[:-12000]

# CNN model specs
model = Sequential([
    Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(56, (3, 3), activation='relu'),
    Flatten(),
    Dense(56, activation='relu'),
    Dense(10, activation='softmax')  # Output layer with 10 nodes for 10 classes
])

# Compile the model with Adam optimizer and sparse categorical cross-entropy loss
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images, train_labels,
    validation_data=(validation_images, validation_labels),
    epochs=10, batch_size=32
)

# Print the number of trainable parameters in the model
print(f"Number of trainable parameters: {model.count_params()}")

import os

# Check for img directory
os.makedirs('imgs', exist_ok=True)


# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.savefig('imgs/training_validation_accuracy.png')  # Save the plot
plt.show()

# Evaluate the model on test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy}")

# Predict labels for test set
predictions = model.predict(test_images)

# Identify misclassified examples
misclassified_idx = np.where(np.argmax(predictions, axis=1) != test_labels)[0]

# Display one misclassified example for each class
for class_id in range(10):
    idx = next((i for i in misclassified_idx if test_labels[i] == class_id), None)
    if idx is not None:
        plt.imshow(test_images[idx].squeeze(), cmap='gray')
        plt.title(f"True: {test_labels[idx]}, Predicted: {np.argmax(predictions[idx])}")
        plt.savefig(f'imgs/misclassified_class_{class_id}.png')  # Save the image
        plt.show()

print("Misclassification examples saved in the imgs/ directory.")
