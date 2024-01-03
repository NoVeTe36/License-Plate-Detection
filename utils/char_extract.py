import cv2
import numpy as np
from matplotlib import pyplot as plt

def char_extract(img, plot_image = False):
  i_h, i_w, _ = img.shape

  min_bound = i_w* i_h *0.02
  max_bound = i_w* i_h * 0.1
  remove_sides = img[i_h//100:i_h - i_h//100, i_w//100: i_w - i_w//100]
  remove_sides_shape = remove_sides.shape

  gray = cv2.cvtColor(remove_sides, cv2.COLOR_RGB2GRAY)



  thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY , 51, 20)
  thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV , 21, 10)

  # Find contours in the binary image
  contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Iterate through contours and cut each character
  cordinates = []
  chars = []
  avg_h = 0
  count = 0
  contours = contours1 if len(contours1) > len(contours2) else contours2
  thresh = thresh1 if len(contours1) > len(contours2) else thresh2

  if plot_image:
    plt.imshow(thresh)
    plt.show()

  for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    contour_area = cv2.contourArea(contour)
    area = w * h
    aspect_ratio = w / h

    min_contour_area = i_h * i_w * 0.03
    max_contour_area =  i_h * i_w * 0.08
    if x <= 2 and h > 30:
      continue

    if y <= 2 and w > 30:

      continue
    if x < 10 and y  < 10:
       continue
    
    if x < 10 and y > remove_sides_shape[0] - 10:
       continue
    
    if x > remove_sides_shape[1] - 10 and y  < 10:
       continue
    
    if y > remove_sides_shape[0] - 10 and x > remove_sides_shape[1] - 10:
       continue
    

    if  (min_bound < area < max_bound) and (0.2 < aspect_ratio < 2):
      character = thresh[y - 1:y + h +1, x - 1:x + w + 1]
      kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
      # character = cv2.filter2D(character, -1, kernel)
      # character = cv2.medianBlur(character, 5)
      avg_h += h
      count += 1
      cordinates.append([x, y])
      chars.append(character)
  

  if count <= 0:
    return None
  avg_h /= count

  points = np.array(cordinates)
  sort_index = np.argsort(points[:, 1])

  lines = [[]]
  current_line = points[sort_index[0]][1]

  for i in sort_index:
    if abs(points[i][1] - current_line) < (avg_h * 0.8):
      lines[-1].append(i)
    else:
      lines.append([i])
      current_line = points[i][1]
    
  final_sort = [sorted(line, key=lambda x: points[x][0]) for line in lines]
  unpack = []
  for line in final_sort:
    unpack += line

  sorted_chars = [chars[i] for i in unpack]

  if plot_image:
    for c in sorted_chars:
      plt.imshow(c)
      plt.show()

  return sorted_chars
  