import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras import backend as K


def VGG_16(height, width, depth, classes):

  model = keras.Sequential()
  inputShape = (height, width, depth)
  chanDim = -1

  if K.image_data_format() == "channels_first":
      inputShape = (depth, height, width)
      chanDim = 1
  
  model = keras.Sequential()
  model.add(ZeroPadding2D((1,1), input_shape=inputShape))
  model.add(Conv2D(32, (3, 3), padding="valid", activation='relu'))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.25))
  
  model.add(ZeroPadding2D((1,1)))
  model.add(Conv2D(64, (3, 3), padding="valid", activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(64, (3, 3), padding="valid", activation='relu'))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  
  model.add(ZeroPadding2D((1,1)))
  model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(ZeroPadding2D((1,1)))
  model.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(ZeroPadding2D((1,1)))
  model.add(Conv2D(512, (3, 3), padding="valid", activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(512, (3, 3), padding="valid", activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(512, (3, 3), padding="valid", activation='relu'))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(classes, activation='softmax'))

  return model


def SmallerVGGNet(width, height, depth, classes):

  model = keras.Sequential()
  inputShape = (height, width, depth)
  chanDim = -1

  if K.image_data_format() == "channels_first":
      inputShape = (depth, height, width)
      chanDim = 1

  model.add(Conv2D(32, (3, 3), padding="valid",input_shape=inputShape))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding="valid"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(64, (3, 3), padding="valid"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, (3, 3), padding="valid"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(128, (3, 3), padding="valid"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(classes))
  model.add(Activation("softmax"))

  return model
