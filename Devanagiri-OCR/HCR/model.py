import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np
# from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from HCR.preprocess import prepareTrainData, prepareValidateData,preparePredictData
from HCR.visualize import predPlot
from HCR.logger import Logger

### Model path
model_path = r"SavedModel/devanagiri.h5"
ckeckpoint_path = r'Checkpoints/cp.ckpt'

### Class CNN Model
class DevnagiriCNN:
    def ___init___(self):
        self.model = None
        self.X_shape = None
        self.Y_shape = None
        self.classes = None
        self.ckpt = None
        Logger.success("DevnagiriCNN Model initialized")
    
    def initModel(self,height,width,channels,classes):
        ### Store input shape and classes
        self.height = height
        self.width = width
        self.channels = channels
        self.classes = classes

        ### CNN Model Architecture ###
        self.model = Sequential()
        self.model.add(Conv2D(filters= 32, kernel_size = (3,3),input_shape = (self.height, self.width, self.channels),padding = 'same',activation = 'relu'))
        self.model.add(BatchNormalization(axis=self.channels))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization(axis=self.channels))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Dense(self.classes, activation = 'sigmoid'))

        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        self.model.summary()

    def train(self,X,Y,epochs = 5,test_size = 0.3,batch_size = 10,model_save = False):
        ### Split data
        x_train ,x_test , y_train, y_test = prepareTrainData(X,Y,self.height,self.width,self.channels,self.classes,test_size=test_size, random_state = 2)

        ### Create checkpoints
        checkpoint1 = ModelCheckpoint(ckeckpoint_path, verbose = 1,save_weights_only=True, save_best_only = True, mode = 'max')
        callbacks_list = [checkpoint1]

        # ### Train CNN model
        self.model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = epochs, batch_size = batch_size, callbacks = callbacks_list)

        # ### Validate Trained model
        val_loss,val_acc = self.model.evaluate(x_test, y_test, verbose = 0)
        Logger.info('Validation Results: val_loss {}  val_acc {}'.format(val_loss,val_acc))

        # ### Save model
        if(model_save == True):
            self.model.save(model_path)
            Logger.info("\nModel saved at {}".format(model_path))
        elif(model_save != False):
            try:
                self.model.save(model_save)
                Logger.success("Model saved at {}".format(model_save))
            except Exception as e:
                Logger.error("Error in filename or path given for saving model....")
                self.model.save(model_path)
                Logger.success("Model saved at {}".format(model_path))
        else:
            pass
    

    def validate(self,X,Y):
        X_valid,Y_valid = prepareValidateData(X,Y,self.height, self.width, self.channels,self.classes)
        val_loss,val_acc = self.model.evaluate(X_valid, Y_valid)
        Logger.info('Validation Results on new Test data : val_loss {}  val_acc: {}'.format(val_loss,val_acc))

        ### Predict with new inputs
        val_predictions = self.model.predict(X_valid)
        print("\n\n")
        

        ### Plotting first 10 inputs with predictions
        for i in range(10):
            predPlot(X[i],self.height,self.width)
            val_prediction = np.argmax(val_predictions[i])
            expected_output = Y_valid[i]
            expected_corresponding_class = np.argmax(Y_valid[i])
            Logger.info("Expected output: {}    Prediction: {}     Expected class: {}".format(expected_output,val_prediction,expected_corresponding_class))



    def predict(self,X):
        ### Prepare predict data
        x_pred = preparePredictData(X,self.height,self.width,self.channels)

        ### Predict with new inputs
        predictions = self.model.predict(x_pred)
        print("\n\n")

        ### Plotting first 10 inputs with predictions
        for i in range(10):
            predPlot(X[i],self.height,self.width)
            prediction = np.argmax(predictions[i])
            Logger.info("Prediction: {}".format(prediction))

        return predictions
    
    def model_config(self,path):
        self.model = tf.keras.models.load_model(path)
        self.model.summary()

    def loadModel(self,height,width,channels,classes,path):
        ### Store input shape and classes
        self.height = height
        self.width = width
        self.channels = channels
        self.classes = classes
        self.model = tf.keras.models.load_model(path)
        Logger.success("Loaded pre-trained model at path: {}".format(path))