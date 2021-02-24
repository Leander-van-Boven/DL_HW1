import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras.utils.np_utils import to_categorical


def get_train_test(dir, validation_split, preprocess):
    X = []
    y = []
    for file in os.listdir(dir):
        x = img_to_array(load_img(os.path.join(dir, file), color_mode='grayscale',
                         target_size=(200, 50)))
        x = preprocess(x)
        X.append(x)
        y.append(file.split('.')[0])
    X = np.array(X)
    y = LabelBinarizer().fit_transform(np.array(y))
    return train_test_split(X, y, test_size=validation_split)
