import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Función de preprocesamiento
def preprocess_captcha_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(eq, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return morph

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
char_to_num = {c: i for i, c in enumerate(alphabet)}
num_to_char = {i: c for i, c in enumerate(alphabet)}

img_width = 200
img_height = 50

def text_to_labels(text):
    text = text.ljust(5, '_')  
    return [char_to_num[c] for c in text]

def preprocess_captcha_image(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return thresh


def load_dataset(folder_path):
    images = []
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            label = os.path.splitext(file)[0].lower()[:5]  # 'abc12.png' → 'abc12'
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            proc = preprocess_captcha_image(img)
            resized = cv2.resize(proc, (img_width, img_height))
            norm = resized.astype(np.float32) / 255.0
            images.append(norm.reshape(img_height, img_width, 1))
            labels.append(text_to_labels(label))

    images = np.array(images)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(alphabet))

    return images, labels

folder_path = 'D:/User/Documents/samples'  
images, labels = load_dataset(folder_path)


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential()

#CNN
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5 * len(alphabet), activation='softmax'))
model.add(Reshape((5, len(alphabet))))


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


CT = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {test_acc * 100:.2f}%")


# Predicciones
predicciones = model.predict(X_test)

predicciones_texto = [''.join([num_to_char[np.argmax(c)] for c in pred]) for pred in predicciones]








def show_predictions(images, labels, predictions, alphabet):
    num_images = len(images)
    
   
    for i in range(min(num_images, 10)):  
        plt.figure(figsize=(2, 2))
        
        
        plt.imshow(images[i].reshape(img_height, img_width), cmap='gray')
        plt.axis('off')  
        
        
        label_text = ''.join([alphabet[np.argmax(c)] for c in labels[i]])
        pred_text = ''.join([alphabet[np.argmax(c)] for c in predictions[i]])
        
        
        plt.title(f"Etiqueta: {label_text}\nPredicción: {pred_text}")
        plt.show()


show_predictions(X_test, y_test, predicciones, alphabet)

