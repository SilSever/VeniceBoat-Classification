from VeniceBoatDataset.dataset_manipulation import read_images
from VeniceBoatDataset.models import VGG_16
from VeniceBoatDataset.metrics import f1, confusion_matrices

import tensorflow as tf
from tensorflow import keras

import matplotlib
 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os


from tensorflow.python.keras.models import load_model
import imutils

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (180, 180, 3) #Height x Width x RGB



#PREPROCESING CNN
data, labels = read_images('VeniceBoatDataset/sc5-tensorflow', 2, IMAGE_DIMS[0], IMAGE_DIMS[1])

class_name = list(set(labels))

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))
 
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")



#TRAINING MODEL

# initialize the model
print("[INFO] compiling model...")
model = SmallerVGGNet(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],depth=IMAGE_DIMS[2], classes=labels.shape[1])
opt = tf.train.AdamOptimizer(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy", f1])

# train the network
print("[INFO] training network...")
H = model.fit_generator( aug.flow(trainX, trainY, batch_size=BS),
                          validation_data=(testX, testY),
                          steps_per_epoch=len(trainX) // BS,
                          epochs=EPOCHS, 
                          verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save('VeniceBoatDataset/model')
 
# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open('VeniceBoatDataset/label', "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.plot(np.arange(0, N), H.history["f1"], label="f1_score")
plt.plot(np.arange(0, N), H.history["val_f1"], label="val_f1_score")
plt.title("Training Accuracy and F1 score")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy/f1")
plt.legend(loc="upper left")
plt.savefig('VeniceBoatDataset/plot')

#plot confusion matrix
cm = metrics.confusion_matrix(model, testY, testX)
metrics.plot_confusion_matrix(cm, class_name)


#TESTING MODEL
# load the image
path = 'VeniceBoat-Dataset/sc5-test-tensorflow/Pleasurecraft/Topa/'
images = os.listdir(path)
im_path = path + images[0]

image = cv2.imread(im_path)
output = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model('VeniceBoat-Dataset/model-family')
lb = pickle.loads(open('VeniceBoat-Dataset/label-family', "rb").read())
 
# classify the input image
print("[INFO] classifying image...")

proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, 'correct')
print("[INFO] {}".format(label))
