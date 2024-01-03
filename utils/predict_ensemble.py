import torch
from torchvision import transforms
from scipy.stats import mode
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

class ResizeMaxSide:
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, img):
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = self.max_size
            new_height = int(self.max_size / aspect_ratio)
        else:
            new_width = int(self.max_size * aspect_ratio)
            new_height = self.max_size


        return transforms.functional.resize(img, (new_height, new_width))

    
class PaddingToSquare:
    def __init__(self):
        pass
    
    def __call__(self, img_pil):
        # Calculate padding to make the image square
        w, h = img_pil.size
        max_size = max(w, h)
        pad_width = max_size - w
        pad_height = max_size - h

        # Apply padding
        padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
        padded_im = transforms.functional.pad(img_pil, padding)
        
        return padded_im
    
class ThresholdTransform(object):
    def __init__(self, thr_255 = None):
        self.thr = thr_255  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        if self.thr is None:
            return (x > torch.mean(x)).to(x.dtype)
        return (x > self.thr).to(x.dtype)  # do not change the data type

transform = transforms.Compose([
    ResizeMaxSide(100),
    PaddingToSquare(),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def predict_chars_ensemble(models, chars):
  x = []
  for c in chars:
    c = Image.fromarray(c)
    x.append(transform(c))
  X = torch.stack(x)
  outputs = []
  for model in models:
    output = model(X)
    outputs.append(torch.argmax(output, dim=1))


  outputs = torch.vstack(outputs)
  out, _ = mode(outputs, 0)
  return out[0]
