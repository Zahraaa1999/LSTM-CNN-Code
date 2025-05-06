

import numpy as np
np.random.seed(1000)

import cv2
import os
from PIL import Image
import keras
from matplotlib import pyplot as plt

plt.rcParams['savefig.dpi'] = 600

os.environ['KERAS_BACKEND'] = 'tensorflow'

# Image Upload
# Import synthtic images to the model


# Model
INPUT_SHAPE = (SIZE, SIZE, 3)
inp = keras.layers.Input(shape=INPUT_SHAPE)

# CNN layers
conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis=-1)(pool1)
drop1 = keras.layers.Dropout(rate=0.2)(norm1)

conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis=-1)(pool2)
drop2 = keras.layers.Dropout(rate=0.2)(norm2)

# Flatten the output from the CNN layers
flat = keras.layers.Flatten()(drop2)

# Reshape to feed into LSTM (batch_size, timesteps, features)
reshaped = keras.layers.Reshape((SIZE, -1))(flat)

# LSTM layer
lstm = keras.layers.LSTM(64, return_sequences=False)(reshaped)

# Fully connected layers
hidden1 = keras.layers.Dense(512, activation='relu')(lstm)
norm3 = keras.layers.BatchNormalization(axis=-1)(hidden1)
drop3 = keras.layers.Dropout(rate=0.2)(norm3)

hidden2 = keras.layers.Dense(256, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis=-1)(hidden2)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)

# Output layer
out = keras.layers.Dense(2, activation='softmax')(drop4)

# Define model
model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Train and Test sets
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size=0.2, random_state=0)

### Fit Model
history = model.fit(np.array(X_train) , y_train , batch_size = 64 , verbose = 1 , epochs = 200 , validation_split = 0.1 , shuffle = False)
print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))


#------------------ Transfer LEarning---------------------

from keras.models import Sequential, Model
from keras import layers

# Creating a new Sequential model to reuse layers from the CNN-LSTM
newmodel = Sequential()

# Adding CNN layers from the original model, excluding the LSTM and fully connected layers
for i, layer in enumerate(model.layers):
    if i < 6:  # Include only the CNN layers before the LSTM
        newmodel.add(layer)

# Freeze the layers from the pre-trained model
for layer in newmodel.layers:
    layer.trainable = False

# Adding new layers for the hybrid model
# Keep the LSTM and fully connected layers trainable

# Add new LSTM layer after the frozen CNN layers
newmodel.add(layers.Reshape((SIZE, -1)))  # Reshape the CNN output to feed into LSTM
newmodel.add(layers.LSTM(64, return_sequences=False))  # Add LSTM layer

# Add new fully connected layers
newmodel.add(layers.Dense(512, activation='relu'))
newmodel.add(layers.BatchNormalization(axis=-1))
newmodel.add(layers.Dropout(rate=0.2))

newmodel.add(layers.Dense(256, activation='relu'))
newmodel.add(layers.BatchNormalization(axis=-1))
newmodel.add(layers.Dropout(rate=0.2))

# Final output layer for classification
newmodel.add(layers.Dense(2, activation='softmax'))


newmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


newmodel.summary()

# Import Transfer Learning images


dataset_learn = np.array(dataset_learn)


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset_learn, to_categorical(np.array(label_learn)), test_size=0.01, shuffle=False)

# Train the new model with transfer learning
history = newmodel.fit(np.array(X_train) , y_train , batch_size = 64 , verbose = 1 , epochs = 200 , validation_split = 0.1 , shuffle = False)
print("Test_Accuracy: {:.2f}%".format(newmodel.evaluate(np.array(X_test), np.array(y_test))[1]*100))
plt.plot(history.history['loss'])
plt.legend(["Training loss"], loc ="upper right")
plt.title('Transfer learning loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(history.history['accuracy'])
#plt.ylim(0.9,1)
plt.legend(["Accuracy"], loc ="lower right")
plt.title('Transfer learning accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


#-------------------Test-------------------

# Import Test Images

# Train-test split
X_train_test, X_test_test, y_train_test, y_test_test = train_test_split(dataset_test, 
                                                                                      to_categorical(np.array(label_test)), 
                                                                                      test_size=0.95, shuffle=False)

# Evaluate the model on test data
print("Test Accuracy: {:.2f}%".format(newmodel.evaluate(np.array(X_test_test), np.array(y_test_test))[1] * 100))
