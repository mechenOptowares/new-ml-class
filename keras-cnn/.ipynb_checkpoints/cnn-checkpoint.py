from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from wandb.keras import WandbCallback
import wandb
import os

run = wandb.init()
config = run.config
config.first_layer_convs = 32
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
# 3x3 kernel
config.dropout = 0.2
config.dense_layer_size = 128
config.img_width = 299
config.img_height = 299
# shape is [28,28,1] =gray image as input for the first convolution
# shape is [28,28,3] = color image

config.epochs = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# reshape input data
X_train = X_train.reshape(
    X_train.shape[0], config.img_width, config.img_height, 1)
X_test = X_test.reshape(
    X_test.shape[0], config.img_width, config.img_height, 1)
# the shape for the X_train [28,28,10]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
labels = range(10)


# build model
model = Sequential()
model.add(Conv2D(32,
                 (config.first_layer_conv_width, config.first_layer_conv_height),
                 input_shape=(28, 28, 1),
                 activation='relu')) # kernel size 3x3 # the output shape after this will be 26x26x32
model.add(MaxPooling2D(pool_size=(2, 2))) # the output shape after this will be 13x13x32
model.add(Conv2D(64,(3,3),
                 activation='relu')) # I am doing the same as line 47 but after pooling, which means I am doing as much features on half of size (because it's after pooling)
model.add(Dropout(0.4)) # Don't droupout after the last layer. Anywhere else will be ok
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Flatten()) # the output shape after this will be 5408
model.add(Dense(config.dense_layer_size, activation='relu')) # the output shape after this will be 5408 x 128 (the layer size)
model.add(Dense(config.dense_layer_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=config.epochs,
          callbacks=[WandbCallback(data_type="image", save_model=False)])
