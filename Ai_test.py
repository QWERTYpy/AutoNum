import keras  # Для создания нейронной сети
from keras.datasets import mnist  # Набор цифр
from keras.models import Sequential  # Простейший тип сети. Вся информация передается только последующему слою
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt  # Для отображения

from keras.layers import BatchNormalization
from keras.layers import Flatten, Conv2D, MaxPooling2D # new!

model = keras.models.load_model('lenet.h5')

path = "./pl_det"
str="0123456789ABCEHKMOPTXY"
for filename in os.listdir(path):

    # grayscale_image = cv2.imread(your_image, 0)
    color_image = cv2.imread(f"{path}/{filename}")
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    n_image = np.divide(gray_image, 255.0)

    plt.imshow(n_image, cmap='Greys')
    plt.show()
    n_image = n_image.reshape(1, 100, 100, 1).astype('float32')
    # print(n_image, type(n_image))

    res = model.predict([n_image]).tolist()
    res = res[0]
    print(f'Это цифра - {str[res.index(max(res))]}')