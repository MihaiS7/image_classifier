# ///////////     DELETING '.DS_Store' file       ///////////
# You can delete it if you do not want to delete this file
import os
try:
    os.system('find . -name .DS_Store | xargs rm')
    print("'.DS_Store' file was completly deleted.")
except Exception as e:
    print("There was an error trying to delete the '.DS_Store' file.")



# 1.1 Install Dependencies and Setup

# 1. Setup and Load Data
# pip install tensorflow tensorflow-gpu opencv-python matplotlib

import tensorflow as tf


# Avoid OOM errors by setting GPU Memory Consumption Growth

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# 1.2 Remove dodgy images
import cv2
from PIL import Image

data_dir = "data"
image_exts = ["jpeg", "jpg", "bmp", "png"]

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        os.system('find . -name .DS_Store | xargs rm')
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = Image.open(image_path)
            if tip.format.lower() not in image_exts:
                print("Image not in ext list {}".format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("Issue with image {}".format(image_path, str(e)))
              
# 1.3 Load Data

import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory(data_dir)

for images, labels in data.take(1):
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(images[:4]):
        ax[idx].imshow(img.numpy().astype(int))
        ax[idx].set_title("Title: {}".format(labels[idx]))



# 2. Preprocess Data
# 2.1 Scale Data

data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

# 2.2 Split Data

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# 3. Deep Model
# 3.1 Build Deep Learning Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3,3), groups=1, activation='relu', input_shape=(256, 256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# 3.2 Train

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# 3.3 Plot Performance

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()



fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='loss')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# 4. Evaluate Performance
# 4.1 Evaluate

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f"Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}")

# 4.2 Test

img_random = cv2.imread("happy.jpeg")
plt.imshow(cv2.cvtColor(img_random, cv2.COLOR_BGR2RGB))
plt.show()

resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

np.expand_dims(resize, 0)
yhat = model.predict(np.expand_dims(resize/255, 0))

print(yhat)
if yhat > 0.5:
    print(f"Predicted class is Sad")
else:
    print(f"Predicted class is Happy")

# 5. Save the Model
# 5.1 Save the Model

from tensorflow.keras.models import load_model

model.save(os.path.join('models', 'happysadmodel.h5'))
new_model = load_model(os.path.join('models', 'happysadmodel.h5'))
new_model.predict(np.expand_dims(resize/255, 0))
