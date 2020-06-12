import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.compat.v1.InteractiveSession(config=config)

model = load_model("cnn.h5")

im = cv2.imread("frame1.jpg")

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im = cv2.resize(im, (28,28))

im = np.reshape(im, (1,28*28))


img = np.argmax(model.predict(im))

print(img)