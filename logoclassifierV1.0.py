import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pathlib
import sys

def filedirinput(s=sys.argv):
    #print ('Number of arguments:', len(sys.argv), 'arguments.')
    #print ('Argument List:', str(sys.argv))
    if len(s) ==2:
        pred_dir = s[1]
    else:
        print("Wrong input. Choose to predict image: ./images/BMW/images (29).jpeg")
        pred_dir = "./images/BMW/images (29).jpeg"
    return pred_dir


def logoclassifier(pred_dir, model, class_names, img_height=180, img_width=180):
    img = keras.preprocessing.image.load_img(pred_dir, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

def main():
    # Image to be claasified
    pred_dir = filedirinput(s=sys.argv)
    # Model input
    img_height = 180
    img_width = 180
    model = tf.keras.models.load_model('./logo_model1')
    with open("class_names.txt", 'r') as f:
        class_names = [line.rstrip('\n') for line in f]
    
    logoclassifier(pred_dir, model, class_names, img_height=img_height, img_width=img_width)

if __name__ == "__main__":
    main()
