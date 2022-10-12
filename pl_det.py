import cv2
import matplotlib as matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import time
import random

# Загружаем обученную модель https://github.com/quangnhat185/Plate_detect_and_recognize
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

# Подготовка изображения
def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

# Создаем список изображений
image_paths = glob.glob("pl_images/*.jpg")
print("Found %i images..."%(len(image_paths)))
"""
# Показываем их
fig = plt.figure(figsize=(12,8))
cols = 3
rows = 2
fig_list = []
for i in range(cols*rows):
    fig_list.append(fig.add_subplot(rows,cols,i+1))
    title = splitext(basename(image_paths[i]))[0]
    fig_list[-1].set_title(title)
    img = preprocess_image(image_paths[i],True)
    plt.axis(False)
    plt.imshow(img)

plt.tight_layout()
plt.show()
"""
# forward image through model and return plate's image and coordinates
# if error "No Licensese plate is founded!" pop up, try to adjust Dmin
def get_plate(image_path, Dmax=608, Dmin=300):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# Получаем изображение номерного знака и его координаты из изборажения
#test_image = image_paths[3]
#test_image = image_paths[4]
#test_image = image_paths[7]


def covert_img(test_image):
    LpImg, cor = get_plate(test_image) # LpImg[0] Получаем изображение номера
    # print("Detect %i plate(s) in"%len(LpImg), splitext(basename(test_image))[0])
    # print("Coordinate of plate(s) in image: \n", cor)

    # Visualize our result
    # plt.figure(figsize=(12,5))
    # plt.subplot(1,2,1)
    # plt.axis(False)
    # plt.imshow(preprocess_image(test_image))
    # plt.subplot(1,2,2)
    # plt.axis(False)
    # plt.imshow(LpImg[0])
    # plt.tight_layout()
    # plt.show()
    #plt.savefig("part1_result.jpg",dpi=300)
    """
    # Рисуем ограничивающую рамку
    def draw_box(image_path, cor, thickness=3):
        pts = []
        x_coordinates = cor[0][0]
        y_coordinates = cor[0][1]
        # store the top-left, top-right, bottom-left, bottom-right
        # of the plate license respectively
        for i in range(4):
            pts.append([int(x_coordinates[i]), int(y_coordinates[i])])
    
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        vehicle_image = preprocess_image(image_path)
    
        cv2.polylines(vehicle_image, [pts], True, (0, 255, 0), thickness)
        return vehicle_image
    
    
    plt.figure(figsize=(8, 8))
    plt.axis(False)
    plt.imshow(draw_box(test_image, cor))
    """

    if (len(LpImg)):  # check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        #print(plate_image.shape)

        # clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(10, 10))
        #
        # lab = cv2.cvtColor(plate_image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        # l, a, b = cv2.split(lab)  # split on 3 different channels
        #
        # l2 = clahe.apply(l)  # apply CLAHE to the L-channel
        #
        # plate_image = cv2.merge((l2, a, b))  # merge channels

        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)


        # Applied inversed thresh_binary
        # binary = cv2.threshold(blur, 180, 255,
        #                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        binary = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 8)
        #return binary
        # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        #return  thre_mor
        # Perform connected components analysis on the thresholded image and
        # initialize the mask to hold only the components we are interested in
        _, labels = cv2.connectedComponents(binary)
        mask = np.zeros(binary.shape, dtype="uint8")
        # Set lower bound and upper bound criteria for characters
        total_pixels = plate_image.shape[0] * plate_image.shape[1] # Разрешение изображения
        lower = 450 #total_pixels // 100  # heuristic param, can be fine tuned if necessary
        upper = 2000 #total_pixels // 20  # heuristic param, can be fine tuned if necessary
        # print(total_pixels,plate_image.shape)
        # Loop over the unique components
        for (i, label) in enumerate(np.unique(labels)):
            # If this is the background label, ignore it
            if label == 0:
                continue

            # Otherwise, construct the label mask to display only connected component
            # for the current label
            labelMask = np.zeros(binary.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            # If the number of pixels in the component is between lower bound and upper bound,
            # add it to our mask
            #print(lower,"<",numPixels, "<", upper)
            if numPixels > lower and numPixels < upper:
                mask = cv2.add(mask, labelMask)
            thre_mor = mask
        return binary, mask
# visualize results
#fig = plt.figure(figsize=(12, 7))
#plt.rcParams.update({"font.size": 18})
#grid = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
#plot_image = [plate_image, gray, blur, binary, thre_mor]
#plot_name = ["plate_image", "gray", "blur", "binary", "dilation"]

# for i in range(len(plot_image)):
#     fig.add_subplot(grid[i])
#     plt.axis(False)
#     plt.title(plot_name[i])
#     if i == 0:
#         plt.imshow(plot_image[i])
#     else:
#         plt.imshow(plot_image[i], cmap="gray")

# plt.savefig("threshding.png", dpi=300)

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def mask_rect(img, cnts):
    img_h = img.shape[0]
    img_w = img.shape[1]
    #print(f"высота-{img_h}, ширина-{img_w}")
    contours_list = []
    # Отсеиваем контуры явно не подходящие под начертания символов
    for c in sort_contours(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        #print(f"x={x}, y={y}, w={w}, h={h}, низ={y + h}, ширина={x + w}, h/w = {h / w}")
        if ratio <= 0.8 or ratio >= 4:
            continue
        contours_list.append((x,y,w,h))
    # Проводим первичную классификацию найденных символов
    list_order_char = [0]*8
    list_order_char_error = [0] * 8
    list_range = [80, 130, 180, 230, 280, 330, 380, 430]
    ideal_contours = [0]*8
    count_range = 0
    for cont in contours_list:
        for lrange in list_range:
            if cont[0] < lrange:
                # Найден первый символ
                if not list_order_char[list_range.index(lrange)]:
                    ideal_contours[list_range.index(lrange)] = cont
                    list_order_char[list_range.index(lrange)] = 1
                    break
                else:
                    list_order_char_error[list_range.index(lrange)] = 1
                    break

    # print(list_order_char)
    # print(list_order_char_error)
    # print(ideal_contours)
    usr_small_h = []
    usr_w = []
    usr_large_h = []
    usr_sdvig = []
    usr_small_y = []
    usr_large_y = []

    for cont in ideal_contours:
        _idx = ideal_contours.index(cont)
        if cont:
            if not list_order_char_error[_idx]:
                if ideal_contours.index(cont) in [0,4,5,6,7]:
                    usr_small_h.append(cont[3])
                    if ideal_contours.index(cont) in [0, 4, 5]:
                        usr_small_y.append(cont[1])
                else:
                    usr_large_h.append(cont[3])
                    usr_large_y.append(cont[1])
                usr_w.append(cont[2])
        if cont:
            if _idx < 7:
                if ideal_contours[_idx+1]:
                    usr_sdvig.append(ideal_contours[_idx+1][0]-ideal_contours[_idx][0])



    usr_small_h = int(sum(usr_small_h)/len(usr_small_h))
    usr_small_y = int(sum(usr_small_y)/len(usr_small_y))
    usr_large_h = int(sum(usr_large_h)/len(usr_large_h))
    usr_large_y = int(sum(usr_large_y)/len(usr_large_y))
    usr_w = int(sum(usr_w)/len(usr_w))
    usr_sdvig = int(sum(usr_sdvig)/len(usr_sdvig))
    # print(usr_small_h,usr_w,usr_large_h, usr_sdvig,usr_small_y,usr_large_y)
    err_flag = False
    while sum(list_order_char) < 8:
        err_flag = not err_flag
        for err_ind in range(len(list_order_char)) if err_flag else range(len(list_order_char)-1, 0, -1):
            if not list_order_char[err_ind]:
                w = usr_w
                if err_ind in [0, 4, 5]:
                    h = usr_small_h
                    y = usr_small_y
                elif err_ind in [6, 7]:
                    h = usr_small_h
                    y = usr_small_y-10

                else:
                    h = usr_large_h
                    y = usr_large_y
                if err_ind < 7 and err_flag:
                    if list_order_char[err_ind+1]:
                        x = ideal_contours[err_ind + 1][0] - usr_sdvig
                        ideal_contours[err_ind] = (x, y, w, h)
                        list_order_char[err_ind] = 1
                if err_ind > 0 and not err_flag:
                    if list_order_char[err_ind-1]:
                        x = ideal_contours[err_ind - 1][0] + usr_sdvig
                        ideal_contours[err_ind] = (x, y, w, h)
                        list_order_char[err_ind] = 1


    for _ind in range(len(list_order_char_error)):
        _y = ideal_contours[_ind][1]
        _w = ideal_contours[_ind][2]
        _h = ideal_contours[_ind][3]

        if list_order_char_error[_ind]:
            if _ind in [0,4,5]:
                if abs(_y - usr_small_y) > 10: _y = usr_small_y
                if abs(_w - usr_w) > 10: _w = usr_w
                if abs(_h - usr_small_h) > 10: _h = usr_small_h
            elif _ind in [6,7]:
                if abs(_y - usr_small_y-10) > 10: _y = usr_small_y
                if abs(_w - usr_w) > 10: _w = usr_w
                if abs(_h - usr_small_h) > 10: _h = usr_small_h
            else:
                if abs(_y - usr_large_y) > 10: _y = usr_large_y
                if abs(_w - usr_w) > 10: _w = usr_w
                if abs(_h - usr_large_h) > 10: _h = usr_large_h
            ideal_contours[_ind]=(ideal_contours[_ind][0],_y,_w,_h)


    for _ind in range(len(ideal_contours)):
        ideal_contours[_ind] = (ideal_contours[_ind][0]-10,ideal_contours[_ind][1]-10,ideal_contours[_ind][2]+20,ideal_contours[_ind][3]+20)









    """
    # Объединяем контуры, которые накладываются друг на друга (для разорванных символов)

    flag_nalogeniya = True
    while flag_nalogeniya:
        for count in range(len(contours_list)-1):
            if contours_list[count+1][0]-contours_list[count][0]-contours_list[count][2] <= 1: # x2-x1+w1
                (x2, y2, w2, h2) = contours_list.pop(count+1)
                (x1, y1, w1, h1) = contours_list.pop(count)
                x = x1
                y = min(y1,y2)
                h = max(h1,h2)
                w = x2+w2-x1
                contours_list.insert(count,(x,y,w,h))
                break
        if count == len(contours_list)-2:
            flag_nalogeniya = False
    #print(f"x={x}, y={y}, w={w}, h={h}, низ={y+h}, ширина={x+w}, h/w = {h/w}")
    
    # Упорядочиваем контуры
    def viborka(list_cont,ind):
        list_ind = []
        for _ in list_cont:
            list_ind.append(_[ind])
        list_ind.remove(max(list_ind))
        list_ind.remove(max(list_ind))
        list_ind.remove(min(list_ind))
        list_ind.remove(min(list_ind))
        sr = 0
        for _ in list_ind:
            sr += _
        sr = int(sr/len(list_ind))
        return sr
    # Ищем среднюю ширину символов
    w_sr = viborka(contours_list,2)
    w_sr_m = w_sr+int(w_sr/2)
    w_sr_b = w_sr+int(w_sr/3)
    # Проверяем первый элемент
    if contours_list[0][0] < 90:
        #Первый элемент найден
        x1 = contours_list[0][0]-int(w_sr/4)
    print(x1)
    y_m = contours_list[0][1]-int(w_sr/4)
    y_b = contours_list[1][1] - int(w_sr / 4)
    h_m = contours_list[0][3]+int(w_sr/2)
    h_b = contours_list[1][3]+int(w_sr/2)

    ideal_contour = [(x1,y_m,w_sr_m,h_m),
                     (x1+w_sr_m,y_b,w_sr_b,h_b),(x1+w_sr_m+w_sr_b,y_b,w_sr_b,h_b),(x1+w_sr_m+2*w_sr_b,y_b,w_sr_b,h_b),
                     (x1+w_sr_m+3*w_sr_b,y_m,w_sr_m-int(w_sr/4),h_m),(x1+2*w_sr_m+3*w_sr_b-int(w_sr/4),y_m,w_sr_m-int(w_sr/4),h_m),
                     (x1+3*w_sr_m+3*w_sr_b-int(w_sr/6),y_m-int(w_sr/4),w_sr_m-int(w_sr/4),h_m),(x1+4*w_sr_m+3*w_sr_b-int(w_sr/2),y_m-int(w_sr/4),w_sr_m-int(w_sr/4),h_m)]

    print(contours_list)
    """
    #return contours_list
    return ideal_contours


for test_image in image_paths:
    try:
        bin_im, msk_im = covert_img(test_image)

        # Find contours and get bounding box for each contour
        cnts, _ = cv2.findContours(msk_im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        #print(boundingBoxes)
        # # Sort the bounding boxes from left to right, top to bottom
        # # sort by Y first, and then sort by X if Ys are similar
        # def compare(rect1, rect2):
        #     if abs(rect1[1] - rect2[1]) > 10:
        #         return rect1[1] - rect2[1]
        #     else:
        #         return rect1[0] - rect2[0]
        #
        #
        # boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))


        for c in mask_rect(msk_im, cnts):
            (x, y, w, h) = c
            if x < 0 : x = 0
            if x > bin_im.shape[1] : x =  bin_im.shape[1]
            if y<0: y=0
            if y>bin_im.shape[0]: y = bin_im.shape[0]

            roi = bin_im[y:y+h, x:x+w]
            #roi = cv2.bitwise_not(roi)
            # plt.imshow(roi)
            # # plt.figure()
            # # plt.imshow(msk_im)
            # plt.tight_layout()
            # plt.show()
            output = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
            # Draw bounding box arroung digit number
            # cv2.rectangle(bin_im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            now_time = str(time.time())[0:10]
            cv2.imwrite(f"./detections/tmp/{now_time}_{random.randint(100, 200)}.png", output)
                    # Sperate number and gibe prediction
                    # curr_num = thre_mor[y:y + h, x:x + w]
                    # curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    # _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # crop_characters.append(curr_num)
        #plt.axis(False)
        cv2.imwrite(f"./detections/tmp/{now_time}_{random.randint(100, 200)}_bin.png", bin_im)
        # plt.figure()
        # plt.imshow(bin_im)
        # #plt.figure()
        # #plt.imshow(msk_im)
        # plt.tight_layout()
        # plt.show()
    except:
        print('Номер не распознан', test_image)




"""
Оригиналяная часть
# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

# define standard width and height of character
digit_w, digit_h = 30, 60

for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if 0.9<=ratio<=3.5: # Only select contour with defined ratio
        if h/plate_image.shape[0]>=0.3: # 0.5 Select contour which has the height larger than 50% of the plate
            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

            # Sperate number and gibe prediction
            curr_num = thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)

print("Detect {} letters...".format(len(crop_characters)))
fig = plt.figure(figsize=(10,6))
plt.axis(False)
plt.imshow(test_roi)
#plt.savefig('grab_digit_contour.png',dpi=300)
plt.tight_layout()
plt.show()
"""