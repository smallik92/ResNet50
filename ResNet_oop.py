import numpy as np
import math
from tensorflow import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
import matplotlib.pyplot as plt

class ResNet():

    def __init__(self, X_train, y_train, X_test, y_test, num_classes, hparameters):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        assert(len(self.X_train)==len(self.y_train))
        assert(len(self.X_test)==len(self.y_test))

        self.num_classes = num_classes
        self.hparameters = hparameters

        (m, n_H0, n_W0, n_C0) = self.X_train.shape
        self.input_shape = (n_H0, n_W0, n_C0)

    def basic_conv_block(self, X, num_filters, stage):

        """
        Implements the following
        Conv2D -> BatchNormalization -> ReLu Activation -> Max Pool
        """

        X = Conv2D(num_filters, (7,7), strides = (2,2), name = 'conv'+str(stage), kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = 'bn'+str(stage))(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3,3), strides = (2,2))(X)

        return X

    def convolutional_block(self, X, num_filters, f, s, stage, block):

        """
        Implements convolutional block of ResNet
        Inputs:
        X: input tensor of shape (m,n_H_prev, n_W_prev, n_C_prev)
        num_filters: list of integer defining the number of filters in the CONV layers of the main path
        f : shape of the middle conv layer's window
        s : stride to be used for first and last conv layers
        stage: integer indicating position in the network
        block: string used to name layers
        """
        conv_name_base = 'res'+str(stage)+block+'_branch'
        bn_name_base = 'bn'+str(stage)+block+'_branch'

        # Retrieve number of filters in each conv layer
        F1, F2, F3 = num_filters

        # Save the input value
        X_shortcut = X

        # Main path
        # First component
        X = Conv2D(F1, (1,1), strides = (s,s),padding = 'valid', name = conv_name_base+'2a', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base+'2a')(X)
        X = Activation('relu')(X)

        #Second component
        X = Conv2D(F2, (f,f), strides = (1,1), padding = 'same', name = conv_name_base+'2b', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
        X = Activation('relu')(X)

        #Third component
        X = Conv2D(F3, (1,1), strides = (1,1), padding = 'valid', name = conv_name_base+'2c', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)

        #Short Cut path
        X_shortcut = Conv2D(F3, (1,1), strides = (s,s), padding = 'valid', name = conv_name_base+'1', kernel_initializer = glorot_uniform(seed = 0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base+'1')(X_shortcut)

        # add the shortcut to the main path
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def identity_block(self, X, num_filters, f, stage, block):

        conv_name_base = 'res'+str(stage)+block+'_branch'
        bn_name_base = 'bn'+str(stage)+block+'_branch'

        # Retrieve number of filters in each conv layer
        F1, F2, F3 = num_filters

        # Save the input value
        X_shortcut = X

        # Main path
        # First component
        X = Conv2D(F1, (1,1), strides = (1,1),padding= 'valid', name = conv_name_base+'2a', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base+'2a')(X)
        X = Activation('relu')(X)

        #Second component
        X = Conv2D(F2, (f,f), strides = (1,1),padding = 'same', name = conv_name_base+'2b', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
        X = Activation('relu')(X)

        #Third component
        X = Conv2D(F3, (1,1), strides = (1,1), padding = 'valid', name = conv_name_base+'2c', kernel_initializer = glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)

        # Add shortcut to the main path
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def resnet_forward(self):

        """
        Implements the ResNet50 architecture
        """
        X_input = Input(self.input_shape)

        #zero-padding
        X = ZeroPadding2D((3,3))(X_input)

        # Stage 1
        X = self.basic_conv_block(X, self.hparameters["S1"]["F1"], 1)

        # Stage 2
        X = self.convolutional_block(X, [self.hparameters["S2"]["conv_F1"], self.hparameters["S2"]["conv_F2"], self.hparameters["S2"]["conv_F3"]],
                                     self.hparameters["S2"]["conv_f"], self.hparameters["S2"]["conv_s"], 2, 'a')
        X = self.identity_block(X, [self.hparameters["S2"]["id_F1"], self.hparameters["S2"]["id_F2"], self.hparameters["S2"]["id_F3"]],
                                     self.hparameters["S2"]["id_f"], 2, 'b')
        X = self.identity_block(X, [self.hparameters["S2"]["id_F1"], self.hparameters["S2"]["id_F2"], self.hparameters["S2"]["id_F3"]],
                                     self.hparameters["S2"]["id_f"], 2, 'c')

        # Stage 3
        X = self.convolutional_block(X, [self.hparameters["S3"]["conv_F1"], self.hparameters["S3"]["conv_F2"], self.hparameters["S3"]["conv_F3"]],
                                     self.hparameters["S3"]["conv_f"], self.hparameters["S3"]["conv_s"], 3, 'a')
        X = self.identity_block(X, [self.hparameters["S3"]["id_F1"], self.hparameters["S3"]["id_F2"], self.hparameters["S3"]["id_F3"]],
                                     self.hparameters["S3"]["id_f"], 3, 'b')
        X = self.identity_block(X, [self.hparameters["S3"]["id_F1"], self.hparameters["S3"]["id_F2"], self.hparameters["S3"]["id_F3"]],
                                     self.hparameters["S3"]["id_f"], 3, 'c')
        X = self.identity_block(X, [self.hparameters["S3"]["id_F1"], self.hparameters["S3"]["id_F2"], self.hparameters["S3"]["id_F3"]],
                                     self.hparameters["S3"]["id_f"], 3, 'd')

        # Stage 4
        X = self.convolutional_block(X, [self.hparameters["S4"]["conv_F1"], self.hparameters["S4"]["conv_F2"], self.hparameters["S4"]["conv_F3"]],
                                     self.hparameters["S4"]["conv_f"], self.hparameters["S4"]["conv_s"], 4, 'a')
        X = self.identity_block(X, [self.hparameters["S4"]["id_F1"], self.hparameters["S4"]["id_F2"], self.hparameters["S4"]["id_F3"]],
                                     self.hparameters["S4"]["id_f"], 4, 'b')
        X = self.identity_block(X, [self.hparameters["S4"]["id_F1"], self.hparameters["S4"]["id_F2"], self.hparameters["S4"]["id_F3"]],
                                     self.hparameters["S4"]["id_f"], 4, 'c')
        X = self.identity_block(X, [self.hparameters["S4"]["id_F1"], self.hparameters["S4"]["id_F2"], self.hparameters["S4"]["id_F3"]],
                                     self.hparameters["S4"]["id_f"], 4, 'd')
        X = self.identity_block(X, [self.hparameters["S4"]["id_F1"], self.hparameters["S4"]["id_F2"], self.hparameters["S4"]["id_F3"]],
                                     self.hparameters["S4"]["id_f"], 4, 'e')
        X = self.identity_block(X, [self.hparameters["S4"]["id_F1"], self.hparameters["S4"]["id_F2"], self.hparameters["S4"]["id_F3"]],
                                     self.hparameters["S4"]["id_f"], 4, 'f')

        # Stage 5
        X = self.convolutional_block(X, [self.hparameters["S5"]["conv_F1"], self.hparameters["S5"]["conv_F2"], self.hparameters["S5"]["conv_F3"]],
                                     self.hparameters["S5"]["conv_f"], self.hparameters["S5"]["conv_s"], 5, 'a')
        X = self.identity_block(X, [self.hparameters["S5"]["id_F1"], self.hparameters["S5"]["id_F2"], self.hparameters["S5"]["id_F3"]],
                                     self.hparameters["S5"]["id_f"], 5, 'b')
        X = self.identity_block(X, [self.hparameters["S5"]["id_F1"], self.hparameters["S5"]["id_F2"], self.hparameters["S5"]["id_F3"]],
                                     self.hparameters["S5"]["id_f"], 5, 'c')

        # Average Pool
        X = AveragePooling2D((2,2), name = 'avg_pool')(X)

        # Output layer
        X = Flatten()(X)
        X = Dense(self.num_classes, activation = 'softmax', name = 'fc'+str(self.num_classes), kernel_initializer= glorot_uniform(seed = 0))(X)

        # create model
        model = Model(inputs = X_input, outputs = X, name = 'ResNet50')

        return model

    def train(self, num_epochs = 70, batch_size = 32, learning_rate = 0.001):

        assert(num_epochs>0 and batch_size>0 and learning_rate>0)

        model = self.resnet_forward()
        opt = keras.optimizers.Adam(learning_rate = 0.001)

        #Train model
        print('Training Model......')
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = model.fit(self.X_train, self.y_train, epochs = num_epochs, batch_size = batch_size)

        return model, history

    def evaluate(self, model):
        preds = model.evaluate(self.X_test, self.y_test)

        return preds[1]
