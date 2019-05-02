from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config 

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
labels = range(10)

num_classes = y_train.shape[1]

# create model
model = Sequential()
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# Don't try to tweek the parameters of the optimizer... too much work....
# categorical_crossentropy should always use if doing multiclass classification. Make the network to output a true probability - image take multiple choice question the student need to assign a probability to the choice, and huge penalty
# for any probability = 0, the result will be any choice will be assigned a non-zero probability regardless how small that is. 
# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=labels, save_model=False)])
