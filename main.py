import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

image_names = list(os.listdir('images/'))

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

def classify_image(image_path, model):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Load the CIFAR-10 model
loaded_model = load_model('my_model.keras')  # Load your trained model here

# Test the classification function
for i in range(len(image_names)):
    image_path = 'images/'+image_names[i]  # Replace with the path to your image
    predicted_class = classify_image(image_path, loaded_model)

    print(f'Class: {class_names[predicted_class]}')
