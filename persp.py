# Тут я попробую разобраться с искажением картинки
import cv2
import numpy as np

img = cv2.imread('detections/5_1664956708_clean.png')
rows,cols,_ = img.shape # Получаем высоту и ширину
cv2.imshow('img',img)
points1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
for count in range(6):
    img_tmp = img.copy()
    if count == 0:
        points2 = np.float32([[0,0],[cols,0+10],[0,rows],[cols,rows-10]])
    if count == 1:
        points2 = np.float32([[0,0],[cols,0+20],[0,rows],[cols,rows-20]])
    if count == 2:
        points2 = np.float32([[0,0+10],[cols,0],[0,rows-10],[cols,rows]])
    if count == 3:
        points2 = np.float32([[0,0+20],[cols,0],[0,rows-20],[cols,rows]])
    if count == 4:
        points2 = np.float32([[0,0],[cols,0],[0+5,rows],[cols-5,rows]])
    if count == 5:
        points2 = np.float32([[0,0],[cols,0],[0+10,rows],[cols-10,rows]])

    matrix = cv2.getPerspectiveTransform(points1,points2)
    # Преобразовать плоскость, состоящую из четырех точек, в плоскость, состоящую из четырех других точек

    output = cv2.warpPerspective(img, matrix, (cols, rows),borderMode=cv2.BORDER_CONSTANT,
      borderValue=(255, 255, 255))
    # Преобразование функцией warpPerspective
    cv2.imshow('output',output)
    cv2.waitKey()
print(rows)
print("=======")
print(cols)
# cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()