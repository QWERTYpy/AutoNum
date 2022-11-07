import os
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import cv2
import numpy as np
import utils
from PIL import Image
import time
import keras


#  Отоброжать только ошибки tensoflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


input_size = 416


def load_images(image_path,model):
    #  Загружаем обученную модель
    saved_model_loaded = tf.saved_model.load("./checkpoints/custom-416", tags=[tag_constants.SERVING])
    #  Загружаем изображение
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    # get image name by using split method
    image_name = image_path.split('/')[-1]
    image_name = image_name.split('.')[0]

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.5
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read in all class names from config
    class_names = utils.read_class_names("./custom.names")

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())

    # custom allowed classes (uncomment line below to allow detections for only people)
    # allowed_classes = ['person']


    image = utils.draw_bbox(model, original_image, pred_bbox, False, allowed_classes=allowed_classes,
                            read_plate=True,)

    image = Image.fromarray(image.astype(np.uint8))
    if not False:
        image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    now_time = str(time.time())[0:10]
    cv2.imwrite(f"./detections/{now_time}.png", image)
# Обработка изображений всех
# for filename in os.listdir("./images"):
#     load_images(f"./images/{filename}")
#     print(filename)
model = keras.models.load_model('lenet.h5')
# Обработка изображений тестовых
for filename in os.listdir("./img_test"):
    load_images(f"./images/{filename}", model)
    print(filename)
