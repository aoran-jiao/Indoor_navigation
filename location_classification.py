for image_batch,label_batch in train_ds:
  pass
base_model = tf.keras.applications.MobileNetV2(input_shape=x_train.shape[1:],
                        include_top=False,
                        weights='imagenet')

feature_batch = base_model(image_batch)
base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
prediction_layer = Dense(7,activation = 'softmax')
prediction_batch = prediction_layer(feature_batch_average)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.compile(optimizer= 'adam', 
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,batch_size = 16, epochs = 10, validation_data=(x_test,y_test))

from tensorflow.keras.layers import *
from tensorflow.keras import Model, Input

tf.keras.applications.InceptionV3(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)
# inputs = Input(shape = X.shape[1:])
# conv1 = Conv2D(32,3,activation='relu')(inputs)
# pool = MaxPool2D(2)(conv1)
# conv2 = Conv2D(32,3,activation='relu')(pool)
# pool = MaxPool2D(2)(conv2)
# conv2 = Conv2D(32,3,activation='relu')(pool)
# pool = MaxPool2D(2)(conv2)
# conv2 = Conv2D(32,3,activation='relu')(pool)
# pool = MaxPool2D(2)(conv2)
# conv2 = Conv2D(32,3,activation='relu')(pool)
# pool = MaxPool2D(2)(conv2)
# conv2 = Conv2D(32,3,activation='relu')(pool)
# pool = MaxPool2D(2)(conv2)
# conv2 = Conv2D(32,3,activation='relu')(pool)
# flat = Flatten()(conv2)
x = Dense(128,activation='relu')(flat)
outputs = Dense(n_classes,activation='softmax')(x)
model = Model(inputs = inputs, outputs = outputs)
model.summary()
model.compile(optimizer= 'adam', 
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(x_train,y_train,batch_size = 16, epochs = 10, validation_data=(x_test,y_test))

model.fit(x_train,y_train,batch_size = 16, epochs = 10, validation_data=(x_test,y_test))

count = 0
for i in range(68):

  s = i
  e = s+1
  if abs(np.argmax(model.predict(x_test[s:e])) - np.argmax(y_test[s:e])) <=1: 
    count+=1
  # print(model.predict(x_test[s:e]))
  # print(np.argmax(model.predict(x_test[s:e])))
  # # print(np.argmax(y_test[s:e]))
  # plt.figure(num=None, figsize=(4,3), dpi=200, edgecolor='k')
  # # plt.imshow()
  # Image.fromarray(((x_test[s]+1)*255).astype(np.uint8),'RGB')
  
model.save(os.path.join(directory,'inference.h5'))

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
