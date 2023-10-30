import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from cnn_basic_model import build_model

def create_spectrogram_model_with_dataset(validation_size=0.15, test_size=0.15):

    np_data_file = "../spectrograms2.npy"
    np_labels_file = "../labels2.npy"

    # get train, validation, test splits
    print("Start loading data...")

    X = np.load(np_data_file)
    y = np.load(np_labels_file)
    y = y[:, 0].astype(np.int64).reshape(-1, 1)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    #  #add an axis to input sets
    # X_train = X_train[..., np.newaxis]
    # X_validation = X_validation[..., np.newaxis]
    # X_test = X_test[..., np.newaxis]

    print(X_train.shape)
    model = build_model((X_train.shape[1],X_train.shape[2], X_train.shape[3]))

    return model, X_train, X_validation, X_test, y_train, y_validation, y_test


def create_model_basic(input_shape):
    # build network topology
    model = keras.Sequential()

     # Convolutional layers suitable for spectrogram data
    model.add(keras.layers.Conv1D(16, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling1D(pool_size=3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))

    # Add more 1D convolutional layers, batch normalization, and dropout as needed
    model.add(keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling1D(pool_size=3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling1D(pool_size=3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling1D(pool_size=3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    # Recurrent layers, e.g., LSTM or GRU
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(128, activation='relu'))
    # output layer
    model.add(keras.layers.Dense(61, activation='softmax'))

    return model


def create_model_cnn_rnn(input_shape):
    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
    model.add(keras.layers.MaxPooling2D((3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # 3rd conv layer
    model.add(keras.layers.Conv2D(64, (3, 3),  activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # 4th conv layer
    model.add(keras.layers.Conv2D(128, (3, 3),  activation='relu', padding='same'))
    model.add(keras.layers.AveragePooling2D((3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Permute((2, 1, 3)))  
    model.add(keras.layers.Reshape((4, 3 * 128)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(128, activation='relu'))

    # output layer
    model.add(keras.layers.Dense(61, activation='softmax'))

    return model
