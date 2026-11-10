import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist


line = "##################################################################"


# Create the neural network

model = tf.keras.Sequential([

    # Turn each 28x28 image into 784 values
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Hidden layer with 64 neurons
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    # Output layer with 10 neurons
    # One neuron for each digit from 0 to 9
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# Show the structure of the model
model.summary()

print(line)


# Compile the model
# ---------------------------------------------------------

# Adam is used to update the model weights
# Categorical crossentropy is used because we have 10 classes
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)



# Load the MNIST dataset

# MNIST already contains training and test data
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()


# Convert labels into categorical form
# Example:
# digit 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
train_Y_categorical = utils.to_categorical(train_Y)
test_Y_categorical = utils.to_categorical(test_Y)


# Print the shapes of the datasets
print("Training images:", train_X.shape)
print("Training labels:", train_Y.shape)

print("Test images:", test_X.shape)
print("Test labels:", test_Y.shape)

print(line)


# ---------------------------------------------------------
# Show one image from the dataset
# ---------------------------------------------------------

plt.imshow(
    train_X[0],
    cmap="gray"
)

plt.title(
    "Label: " + str(train_Y[0])
)

plt.show()


# ---------------------------------------------------------
# Train the neural network
# ---------------------------------------------------------

# Epochs = how many times the model sees the training data
# 33% of the training data is used for validation
history = model.fit(
    train_X,
    train_Y_categorical,
    epochs=10,
    validation_split=0.33
)


print(line)


# ---------------------------------------------------------
# Plot training and validation accuracy
# ---------------------------------------------------------

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# ---------------------------------------------------------
# Plot training and validation loss
# ---------------------------------------------------------

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


# ---------------------------------------------------------
# Make a prediction on the first test image
# ---------------------------------------------------------

prediction = model.predict(
    test_X[0:1]
)


# Print the 10 output values
print("Prediction values:")
print(prediction[0])


# Find the digit with the highest value
predicted_digit = np.argmax(
    prediction[0]
)


print("Predicted digit:", predicted_digit)
print("Actual digit:", test_Y[0])

print(line)


# ---------------------------------------------------------
# Show the first test image
# ---------------------------------------------------------

plt.imshow(
    test_X[0],
    cmap="gray"
)

plt.title(
    "Prediction: "
    + str(predicted_digit)
    + " | Actual: "
    + str(test_Y[0])
)

plt.show()


# ---------------------------------------------------------
# Make predictions for all test images
# ---------------------------------------------------------

test_predictions = model.predict(
    test_X
)


# Convert the 10 output values into one predicted digit
predicted_labels = np.argmax(
    test_predictions,
    axis=1
)


# ---------------------------------------------------------
# Create the confusion matrix
# ---------------------------------------------------------

confusion_matrix = tf.math.confusion_matrix(
    labels=test_Y,
    predictions=predicted_labels
)


print("Confusion Matrix:")
print(confusion_matrix)

print(line)


# ---------------------------------------------------------
# Evaluate the model on the test data
# ---------------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    test_X,
    test_Y_categorical
)


print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

print(line)


# ---------------------------------------------------------
# Access the hidden layer
# ---------------------------------------------------------

# Get the individual layers from the model
flatten_layer = model.layers[0]
hidden_layer = model.layers[1]
output_layer = model.layers[2]


# Take one test image
image = test_X[0]


# Add one extra dimension because the model expects a batch
image_tensor = tf.convert_to_tensor(
    np.reshape(image, (1, 28, 28)),
    dtype=tf.float32
)


# Pass the image through the flatten layer
flattened_image = flatten_layer(
    image_tensor
)


# Pass the flattened image through the hidden layer
hidden_output = hidden_layer(
    flattened_image
)


# Pass the hidden layer output to the final layer
output = output_layer(
    hidden_output
)


# Get the final predicted digit
hidden_prediction = np.argmax(
    output.numpy()[0]
)


print("Prediction from hidden-layer process:")
print(hidden_prediction)


# ---------------------------------------------------------
# Show the hidden layer output
# ---------------------------------------------------------

# The hidden layer has 64 values
# 64 values can be reshaped into an 8x8 image
hidden_image = hidden_output.numpy().reshape(
    8,
    8
)


plt.imshow(
    hidden_image,
    cmap="gray"
)

plt.title(
    "Hidden Layer Output"
)

plt.show()


print(line)