import math
import os
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import pandas as pd

#image preprocessing and convert image into tensor
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return x

#extract image feature function
def extract_vector(model, image_path):
    print("Processing: ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)
    #extract feature
    vector = model.predict(img_tensor)[0]
    #Normalize vector 
    vector = vector / np.linalg.norm(vector)
    return vector

#Create model function
def get_extract_model():
    vgg16_model = VGG16(weights = "imagenet")
    extract_model = Model(inputs = vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
    return extract_model

#define data folder
data_folder = './static/image/DATA3/img_data'

#Initialize model 
model = get_extract_model()
vectors = []
paths = []
contents = []

for folder_name in os.listdir(data_folder):
    folder_path_full = os.path.join(data_folder, folder_name)
    for image_path in os.listdir(folder_path_full):
        content = str(image_path[0:3])
        image_path_full = os.path.join(data_folder, folder_name, image_path)
        image_vector = extract_vector(model, image_path_full)
        vectors.append(image_vector)
        paths.append(image_path_full)
        contents.append(content)

#Create the dataframe
df = pd.DataFrame(np.array(vectors))
df['Content'] = pd.Series(contents, index = df.index)
df['Path'] = pd.Series(paths, index = df.index)

df.to_csv("./static/feature/features.csv", index = False)


