from PIL import Image
import pytesseract
import cv2
import os

image = 'Num/111.jpg'
#image = 'Num/321.png'
preprocess = "thresh"
preprocess = "blur"

# загрузить образ и преобразовать его в оттенки серого
image = cv2.imread(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# проверьте, следует ли применять пороговое значение для предварительной обработки изображения

if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# если нужно медианное размытие, чтобы удалить шум
elif preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)
# показать выходные изображения
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey()
# сохраним временную картинку в оттенках серого, чтобы можно было применить к ней OCR

filename = "{}.jpg".format(os.getpid())
cv2.imwrite(filename, gray)
#filename = image
def build_tesseract_options(psm=7):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABEHKMOPTXYacxopky0123456789"
    #alphanumeric = "АВЕКМНОРСТУХ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    options += " --psm {}".format(psm)
    # return the built options string
    return options
# загрузка изображения в виде объекта image Pillow, применение OCR, а затем удаление временного файла
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
options = build_tesseract_options(psm=7)

text = pytesseract.image_to_string(Image.open(filename), config=options)
os.remove(filename)
print("==>",text)

# показать выходные изображения
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
input('pause…')