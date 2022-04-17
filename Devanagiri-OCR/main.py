import pandas as pd
from HCR.preprocess import prepareInputData
from HCR.model import DevnagiriCNN
from HCR.visualize import plot
from HCR.logger import Logger

#dataset and model path
TRAIN_DATA_PATH = r'./Dataset/Devnagiri/train.csv'
TEST_DATA_PATH = r'./Dataset/Devnagiri/test.csv'
MODEL_PATH = r"./SavedModel/devanagiri.h5"

### Parameters
HEIGHT = 32
WIDTH = 32
CHANNELS = 1
NUM_CLASSES = 10

#Hyperparameters
EPOCHS = 20
TEST_SIZE=0.1
BATCH_SIZE = 64



def train():
    ### Load Dataset ###
    train_data = pd.read_csv(TRAIN_DATA_PATH,header=None)
    test_data = pd.read_csv(TEST_DATA_PATH,header=None)

    ### Prepare Dataset in the form of X and Y values
    """Training data"""
    X_train,Y_train = prepareInputData(train_data)

    """Testing data"""
    X_test,Y_test = prepareInputData(test_data)

    ### Visualize
    plot(X_train[0],Y_train[0],HEIGHT,WIDTH)


    ### initialize model pipeline
    model = DevnagiriCNN()

    ### initialize model pipeline
    model.initModel(
        height=HEIGHT,
        width=WIDTH,
        channels=CHANNELS,
        classes=NUM_CLASSES
        )


    ### Train model 
    model.train(
        X=X_train,
        Y=Y_train,
        epochs=EPOCHS,
        test_size=TEST_SIZE,
        batch_size=BATCH_SIZE,
        model_save=True
        )
    

    ### Validate model
    model.validate(
        X=X_test,
        Y=Y_test
        )



def test():
    ### Load Dataset ###
    test_data = pd.read_csv(TEST_DATA_PATH,header=None)

    ### Prepare Dataset in the form of X and Y values
    """Testing data"""
    X_test,Y_test = prepareInputData(test_data)

    ### initialize model pipeline
    model = DevnagiriCNN()

    ### Load CNN model
    model.loadModel(
        height=HEIGHT,
        width=WIDTH,
        channels=CHANNELS,
        classes=NUM_CLASSES,
        path=MODEL_PATH
        )

    ### Predict Model 
    model.predict(
        X=X_test
        )




def main():
    Logger.init()
    inp_option = Logger.menu()
    try:
        if inp_option == "q" or inp_option=="Q":
            Logger.success("Exiting the application")

        elif float(inp_option) == 1:
            train()
        
        elif float(inp_option) == 2:
            test()

    except Exception as e:
        Logger.error(e)


if __name__ == '__main__':
    main()
