from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.image as gimage
import os
import numpy as np
from PIL import Image

class BModel():

    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(416, 416, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

    def train(self, x_train, y_train, batch_size=128, epochs=50, log_dir='./'):
        logging = TensorBoard(log_dir=log_dir)
        checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                     monitor='val_loss', save_weights_only=True, save_best_only=True, period=2)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.fit(x=x_train, y=y_train, validation_split=0.2,
                       epochs=100, batch_size=128, callbacks=[logging, checkpoint, early_stopping, reduce_lr])
        self.model.save_weights('weights.h5')

    def save(self, file_path):
        print('Model Saved.')
        self.model.save_weights(file_path)

    def load(self, file_path):
        print('Model Loaded.')
        self.model.load_weights(file_path)

    def predict(self, image):
        #img = gimage.imread(image)
        #img = Image.open(image)
        img = image.resize((416,416))
        img  = np.array(img)
        im = np.zeros((1, 416, 416, 3))
        im[0, :, :, :] = img / 255

        # 归一化
        #result = self.model.predict(im)
        #print(result)
        # 概率
        result = self.model.predict_classes(im)
        #print(result)
        # 0/1

        return result

    def evaluate(self, dataset):
        # 测试样本准确率
        score = self.model.evaluate_generator(dataset.valid, steps=2)
        print("样本准确率%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
