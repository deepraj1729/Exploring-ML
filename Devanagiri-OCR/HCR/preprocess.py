import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import normalize
from keras.utils import np_utils

def prepareInputData(data):
    dataset  = np.array(data)
    # dataset = np.random.shuffle(dataset)
    
    rows,cols = dataset.shape
    
    ### Extracting X in shape (:,1:cols) i.e. except 1st column
    X= dataset[:, 1:cols]

    ### Normalizing X
    X = normalize(X, axis=1)
    # X = X/255

    # Extracting Y as the first column
    Y = dataset[:, 0]
    
    ### Resizing Y in the form (len(Y),1)
    Y = Y.reshape(Y.shape[0], 1)
    return X,Y



### Preparing data for training and testing
def prepareTrainData(X,Y,height,width,channels,classes,test_size,random_state):
    ### Resize X 
    ### Format: (len(X), height, width, channels)
    X = X.reshape(X.shape[0], height, width, channels)

    ### Y to categorical i.e 1 -> [1,0,0,0,0] for 5 classes
    Y = np_utils.to_categorical(Y,classes)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = test_size, random_state = random_state)
    return X_train,X_test,Y_train,Y_test

def prepareValidateData(X,Y,height, width, channels,classes):
    ### Resize X 
    ### Format: (len(X), height, width, channels)
    X_valid = X.reshape(X.shape[0], height, width, channels)

    ### Y to categorical i.e 1 -> [1,0,0,0,0] for 5 classes
    Y_valid = np_utils.to_categorical(Y,classes)
    return X_valid,Y_valid

def preparePredictData(X,height, width, channels):
    X_pred = X.reshape(X.shape[0], height, width, channels)
    return X_pred