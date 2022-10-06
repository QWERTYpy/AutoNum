# Добавляет к каждой картинке по 6 искажений в перспективе.
import cv2
import numpy as np
import os



def convert(path, filename):
    img = cv2.imread(f'{path}/{filename}')
    rows,cols,_ = img.shape # Получаем высоту и ширину
    #cv2.imshow('img',img)
    points1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
    for count in range(6):
        nname=""
        img_tmp = img.copy()
        if count == 0:
            nname="r"
            points2 = np.float32([[0,0],[cols,0+10],[0,rows],[cols,rows-10]])
        if count == 1:
            nname = "rr"
            points2 = np.float32([[0,0],[cols,0+20],[0,rows],[cols,rows-20]])
        if count == 2:
            nname = "l"
            points2 = np.float32([[0,0+10],[cols,0],[0,rows-10],[cols,rows]])
        if count == 3:
            nname = "ll"
            points2 = np.float32([[0,0+20],[cols,0],[0,rows-20],[cols,rows]])
        if count == 4:
            nname = "b"
            points2 = np.float32([[0,0],[cols,0],[0+5,rows],[cols-5,rows]])
        if count == 5:
            nname = "bb"
            points2 = np.float32([[0,0],[cols,0],[0+10,rows],[cols-10,rows]])

        matrix = cv2.getPerspectiveTransform(points1,points2)
        # Преобразовать плоскость, состоящую из четырех точек, в плоскость, состоящую из четырех других точек

        output = cv2.warpPerspective(img, matrix, (cols, rows),borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255))
        output = cv2.cvtColor(np.array(output), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{path}/{nname}_{filename}", output)
        #output = cv2.resize(output,(100,100), interpolation = cv2.INTER_AREA)
        # Преобразование функцией warpPerspective
    # cv2.imshow('output',output)
    # cv2.waitKey()
# convert("./train/0","0_1664966280_clean.png")
path = "./train"
for dirname in os.listdir(path):
    for filename in os.listdir(f"{path}/{dirname}"):
        convert(f"{path}/{dirname}", filename)
