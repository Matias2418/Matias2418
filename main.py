import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow_hub as hub 
import datetime
from PIL import Image
import cv2
import base64
from typing import Union
import io
from fastapi import FastAPI, File, UploadFile

modelt = load_model(('model_mobilenetv2.h5'), custom_objects={'KerasLayer':hub.KerasLayer})
class_names = ['ANORMALES', 'HISTORIAL', 'MIOCARDIO', 'NORMAL']
width_shape = 224
height_shape = 224
app = FastAPI()

@app.post("/")
def read_root(data):
    image = Image.open(io.BytesIO(base64.b64decode(data)))
    image = np.array(image).astype(float)/255
    image = cv2.resize(image, (224,224))
    prediccion = modelt.predict(image.reshape(-1, 224, 224, 3))
    prediccion2= np.argmax(prediccion[0], axis=-1)
    return {"arritmia":class_names[prediccion2]}