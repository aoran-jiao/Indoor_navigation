try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
  
  
from google.colab import drive
drive.mount('/content/drive')

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

!pip install pyheif
!pip install whatimage

import whatimage
import pyheif
from PIL import Image


def decodeImage(bytesIo):

    fmt = whatimage.identify_image(bytesIo)
    if fmt in ['heic', 'avif']:
            i = pyheif.read_heif(bytesIo)

            # Extract metadata etc
            for metadata in i.metadata or []:
                if metadata['type']=='Exif':
                    # do whatever

            # Convert to other file format like jpeg
                    s = io.BytesIO()
                    pi = Image.frombytes(
                        mode=i.mode, size=i.size, data=i.data)

                    pi.save(s, format="jpeg")
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

X = []
y = []
directory = '/content/drive/My Drive/New_Colab/Myhal photos'

ind = 1
n_classes = 7

for subdir in os.listdir(directory):
  dirpath = os.path.join(directory,subdir)
  count = 0
  for f in os.listdir(dirpath):
    count+=1
    filename = os.path.join(dirpath,f)
    if filename.endswith('.HEIC'):
      bytesIo = filename
      # fmt = whatimage.identify_image(bytesIo)
      # if fmt in ['heic', 'avif']:
      i = pyheif.read_heif(bytesIo)
      # Extract metadata etc
      for metadata in i.metadata or []:
        if metadata['type']=='Exif':

          pi = Image.frombytes(mode=i.mode, size=i.size, data=i.data)
          pi = pi.resize([480,640])
          if count == ind:
            plt.figure(num=None, figsize=(4,3), dpi=200, edgecolor='k')
            plt.imshow(pi)
          pi = np.array(pi)
          pi = pi/127.5-1
          pii = Image.fromarray(((pi+1)*127.5).astype(np.uint8),'RGB')
          if count == ind:
            plt.figure(num=None, figsize=(4,3), dpi=200, edgecolor='k')
            plt.imshow(pii)
          X.append(pi)
          lab = tf.one_hot(int(subdir),depth = n_classes).numpy()
          y.append(lab)
          print(lab)
      if count == ind:
        break  

  break

    #     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"): 
    #         img = cv2.imread(filename)
    #         img = cv2.resize(img,image_shape)
    #         X.append(img)
    #         y.append(tf.one_hot(int(subdir)).numpy())
    #     else:
    #         continue
X = np.array(X)
y = np.array(y)

