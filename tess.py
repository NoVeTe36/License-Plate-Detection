from PIL import Image

import cv2
import pytesseract
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'D:\\tesseract\\tesseract.exe'

plate = cv2.imread("plate.jpg")
cropped = plate
# cropped = plate[10:-10, 30:-30]
# t, cropped = cv2.threshold(cropped, 100, 255, cv2.THRESH_BINARY)

plt.imshow(cropped)
plt.show()
config = "--psm 11"
s = pytesseract.image_to_string(cropped, config=config)
print(s)
