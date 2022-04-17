import matplotlib.pyplot as plt
from HCR.logger import Logger

def plot(X,Y,height,width):
    z = X.reshape(height,width)
    plt.imshow(z,cmap = plt.cm.binary)
    plt.show()
    Logger.info("Class label: {}".format(Y))


def predPlot(X,height,width):
    z = X.reshape(height,width)
    plt.imshow(z,cmap = plt.cm.binary)
    plt.show()