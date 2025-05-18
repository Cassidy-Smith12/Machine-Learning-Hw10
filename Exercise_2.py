from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import cv2
import numpy as np
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1).astype("float32") / 255
X_test = X_test.reshape(10000, 28, 28, 1).astype("float32") / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
cnn.add(Flatten())
cnn.add(Dense(units=256, activation="relu"))
cnn.add(Dense(units=10, activation="softmax"))

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
cnn.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    image = cv2.resize(image, (28, 28))  
    image = image.astype("float32") / 255  
    image = image.reshape((1, 28, 28, 1))  
    return image

def predict_digit(cnn, image_path):
    image = preprocess_image(image_path)
    prediction = cnn.predict(image)
    return np.argmax(prediction), prediction

image_paths = ['two.png', 'three.png'] 

print("Predictions for handwritten digit images:")
for image_path in image_paths:
    digit, probabilities = predict_digit(cnn, image_path)
    print(f"Prediction for {image_path}: {digit}")
    plt.imshow(cv2.imread(image_path))
    plt.title(f'Predicted Digit: {digit}')
    plt.axis('off')  
    plt.show()
