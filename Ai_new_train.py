import keras  # Для создания нейронной сети
from keras.datasets import mnist  # Набор цифр
from keras.models import Sequential  # Простейший тип сети. Вся информация передается только последующему слою
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt  # Для отображения

from keras.layers import BatchNormalization
from keras.layers import Flatten, Conv2D, MaxPooling2D # new!

data_dir = "./tmp"
img_height = 100
img_width = 100
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  label_mode='categorical',
  validation_split=0.2,
  subset="training",
  color_mode = "grayscale",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# Вывод данных о изображениях
#for x, y in train_ds.take(1):
#  print(x.shape,"===", y)
# Вывод имен классов
# class_names = train_ds.class_names
# print(class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  label_mode='categorical',
  validation_split=0.2,
  subset="validation",
  color_mode = "grayscale",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

print(train_ds)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_train_ds))
# first_image = image_batch[0]
# # Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))
#
n_classes = 24
# normalized_train_ds = to_categorical(normalized_train_ds, n_classes)
# normalized_val_ds = to_categorical(normalized_val_ds, n_classes)


model = Sequential()
# первый сверточный слой:
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 1)))

# второй сверточный слой с субдискретизацией и прореживанием:
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # преобразует трехмерную карту активаций, сгенерированную слоем Conv2D(), в одномерный массив

# полносвязанный скрытый слой с прореживанием:
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# выходной слой:
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_valid, y_valid))
history = model.fit(normalized_train_ds, epochs=10, validation_data=normalized_val_ds)
model.save('lenet_new.h5')

