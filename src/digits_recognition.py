import os
import json
import keras
import time
import tensorflow as tf
import numpy as np

from prepare_dataset import prepare_datasets
from plot_history import plot_history
from sklearn.model_selection import train_test_split

def build_model_4l(input_shape):
    """Generates CNN model

    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
     # 4th conv layer
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.6))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def prepare_datasets(data_path, test_size, validation_size):
    
    with open(data_path, "r") as json_file:
        dataset_dict = json.load(json_file)

    # Convert the lists in the dictionary back to NumPy arrays
    data = np.array(dataset_dict["data"])
    labels = np.array(dataset_dict["labels"])

    labels = labels[:, 1].astype(np.int64).reshape(-1, 1)
    
    X, y = data, labels


    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the models folder
models_folder = os.path.join(script_dir, '..', 'models')

old_data = "../data.json"

# get train, validation, test splits
print("Start loading data...")
X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(old_data, 0.15, 0.15)

input_shape = (X_train.shape[1], X_train.shape[2], 1)

model = build_model_4l(input_shape)

# compile model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

model.summary()

# Start measuring time
start_time = time.time()

# train model
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=20)

# Stop measuring time
end_time = time.time()

# Calculate the training time in seconds
training_time_seconds = end_time - start_time

# Convert training time to hours, minutes, and seconds
training_hours, remainder = divmod(training_time_seconds, 3600)
training_minutes, training_seconds = divmod(remainder, 60)


# Print the training time in a human-readable format
print(f"Training time: {int(training_hours):02d} hours, {int(training_minutes):02d} minutes, {int(training_seconds):02d} seconds")

# Define the relative path to the models folder
models_folder = os.path.join(script_dir, '..', 'models')

# plot accuracy/error for training and validation
plot_history(history, os.path.join(models_folder, 'train_history_mfcc_digits_model_4l_60e_04d_2'))

# evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model to the models folder
model.save(os.path.join(models_folder, 'mfcc_digits_model_4l_60e_04d_2.h5'))

with open('training_history_mfcc_digits_model_4l_60e_04d_2.json', 'w') as json_file:
    json.dump(history.history, json_file)