import matplotlib.pyplot as plt

def plotHistory(history):
    """Plot training/validation accuracy and loss. history is Callback obj from Keras
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.legend(['train','val'],loc='lower right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.legend(['train','val'],loc='upper right')
    plt.show()

def aug_compare(image, augFun, **kwargs):
    '''plot original and augmented image
       augFun take an image and return an transformed image
    '''
    f, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))
    ax[0].imshow(image,**kwargs)
    ax[1].imshow(augFun(image),**kwargs)
    plt.show()