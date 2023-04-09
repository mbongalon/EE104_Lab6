# import the necessary modules
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
from keras.optimizers import RMSprop
import numpy as np

# load the train and test dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# convert from integer type to float32
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# normalize the pixel values
mean = np.mean(train_images,axis=(0,1,2,3))
std = np.std(train_images,axis=(0,1,2,3))
train_images = (train_images-mean)/(std+1e-7)
test_images = (test_images-mean)/(std+1e-7)

# convert to one-hot encoded vectors
num_classes = 10
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

# set the default number of convolutional filters and weight decay
baseMapNum = 32
weight_decay = 1e-4

# define the convolutional neural network (cnn) model
model = Sequential()
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=train_images.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# connect the different layers
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(train_images)

# set the default number of samples and epochs
batch_size = 64
epochs = 25

# train the model
opt_rms = RMSprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),steps_per_epoch=train_images.shape[0] // batch_size,epochs=3*epochs,verbose=1,validation_data=(test_images,test_labels))
model.save('cifar10_normal_rms_ep75.h5')

opt_rms = RMSprop(lr=0.0005,decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),steps_per_epoch=train_images.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(test_images,test_labels))
model.save('cifar10_normal_rms_ep100.h5')

opt_rms = RMSprop(lr=0.0003,decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),steps_per_epoch=train_images.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(test_images,test_labels))
model.save('cifar10_normal_rms_ep125.h5')

# test the model
scores = model.evaluate(test_images, test_labels, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))