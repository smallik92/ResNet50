import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ResNet_oop import ResNet
import h5py

np.random.seed(1)

def load_dataset():
    train_dataset = h5py.File('train_signs.h5',"r")
    test_dataset = h5py.File('test_signs.h5',"r")

    train_set_x = np.array(train_dataset["train_set_x"][:]) #train set features
    train_set_y = np.array(train_dataset["train_set_y"][:]) #train set labels

    test_set_x = np.array(test_dataset["test_set_x"][:]) #test set features
    test_set_y = np.array(test_dataset["test_set_y"][:]) #test set labels

    classes = np.array(test_dataset["list_classes"][:]) #list of classes

    return train_set_x, train_set_y, test_set_x, test_set_y, classes

def create_one_hot(Y,C):
    """
    Y: labels
    C: number of classes
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def main():
    X_train, y_train, X_test, y_test, classes = load_dataset()
    X_train = X_train/255
    X_test = X_test/255

    num_classes = len(classes)
    y_train = create_one_hot(y_train, num_classes).T
    y_test = create_one_hot(y_test, num_classes).T

    hparameters = {'S1': {'F1':64 },
                   'S2': {'conv_F1':64, 'conv_F2':64, 'conv_F3':256, 'conv_f':3, 'conv_s':1, 'id_F1':64, 'id_F2':64, 'id_F3':256, 'id_f':3},
                   'S3': {'conv_F1':128, 'conv_F2':128, 'conv_F3':512, 'conv_f':3 , 'conv_s':2,'id_F1':128, 'id_F2':128, 'id_F3':512, 'id_f':3},
                   'S4': {'conv_F1':256, 'conv_F2':256 , 'conv_F3':1024 , 'conv_f':3 , 'conv_s':2, 'id_F1':256, 'id_F2':256, 'id_F3':1024, 'id_f':3},
                   'S5': {'conv_F1':512, 'conv_F2':512, 'conv_F3':2048, 'conv_f':3, 'conv_s':2, 'id_F1':512, 'id_F2':512, 'id_F3':2048, 'id_f':3}}

    resnet50_network = ResNet(X_train, y_train, X_test, y_test, num_classes, hparameters)
    model, history = resnet50_network.train(num_epochs = 20, batch_size = 32, learning_rate = 0.001)
    test_accuracy = resnet50_network.evaluate(model)

    print('Accuracy on test set: {:.3f} %'.format(test_accuracy*100))

    plt.plot(history.history['loss'])
    plt.title('Training error')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()

main()
