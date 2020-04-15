#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os 
import numpy as np 
import pandas as pd
from datetime import datetime
import argparse

from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, SpatialDropout2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.regularizers import l2

from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder

keras_backend.set_image_data_format('channels_last')

"""
CNN model fitting
"""
def create_model(input_shape, spatial_dropout_rate_1=0, spatial_dropout_rate_2=0, l2_rate=0):

    # Create a secquential object
    model = Sequential()

    for i in range(2):
        model.add(Conv2D(filters=32, 
                         kernel_size=(3, 3), 
                         kernel_regularizer=l2(l2_rate), 
                         input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(SpatialDropout2D(spatial_dropout_rate_1))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    for i in range(2):
        model.add(Conv2D(filters=64, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(SpatialDropout2D(spatial_dropout_rate_2))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    for i in range(2):
        model.add(Conv2D(filters=128, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(SpatialDropout2D(spatial_dropout_rate_2))
    
    # Reduces each h×w feature map to a single number by taking the average of all h,w values.
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    # Softmax output
    model.add(Dense(10, activation='softmax'))
    
    return model




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_feature", type = str, default = "/efs/kevin/audio/features/X_mfcc_80_train_aug.pickle", help ="Pickle file for training feature")
    parser.add_argument("--train_label", type = str, default = "/efs/kevin/audio/features/Y_train_aug.pickle", help ="Pickle file for training label")
    parser.add_argument("--epochs", type = int, default = 300, help ="number of epochs")
    parser.add_argument("--batch_size", type = int, default = 128, help ="batch size")
    parser.add_argument("--spatial_do_1", type = float, default = 0.1, help ="spatial dropout 1")
    parser.add_argument("--spatial_do_2", type = float, default = 0.2, help ="spatial dropout 2")
    parser.add_argument("--l2_rate", type = float, default = 0.1, help ="l2 regularization")
    parser.add_argument("--model", type = str, default = '/efs/kevin/audio/models/model.hdf5', help ="l2 regularization")
    
    
    opt = parser.parse_args()
    print(opt)

    """
    Convert the format of X and y 
    """
    le = LabelEncoder()
    X = pickle.load(open(opt.train_feature, "rb" ) )
    y = pickle.load(open(opt.train_lable, "rb" ) )


    X_train = np.array(X) 
    X = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    y_train_encoded = to_categorical(le.fit_transform(y))
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    """
    Set model parameter 
    """

    model = create_model(input_shape, opt.spatial_do_1, opt.spatial_do_2, opt.l2_rate)


    adam = Adam(lr=3e-5, decay=1e-6, beta_1=0.90, beta_2=0.99)
    model.compile(
        loss='categorical_crossentropy', 
        metrics=['accuracy'], 
        optimizer=adam)
    model.summary()

    num_epochs = opt.epochs
    num_batch_size = opt.batch_size
    
    # Save checkpoints
    checkpointer = ModelCheckpoint(filepath= opt.model, 
                                   verbose=1, 
                                   save_best_only=True)
    start = datetime.now()
    history = model.fit(X, 
                        y_train_encoded, 
                        batch_size=num_batch_size, 
                        epochs=num_epochs, 
                        validation_split=1/12.,
                        callbacks=[checkpointer], 
                        shuffle = True,
                        verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

