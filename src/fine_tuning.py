import os
import json
import keras
import time
import tensorflow as tf
import numpy as np

from prepare_dataset import prepare_datasets
from plot_history import plot_history


# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the models folder
models_folder = os.path.join(script_dir, '..', 'models')

# Load the pre-trained model
pretrained_model = tf.keras.models.load_model(os.path.join(models_folder, 'mfcc_model_4l_60e_04d_2.h5'))

# Define the new output layer with one additional unit for the new users
new_output_layer = tf.keras.layers.Dense(67, activation='softmax')

old_data = "../data_sr_48k.json"
new_data = "../data_new_users_sr_48k.json"

# get train, validation, test splits
print("Start loading data...")
accuracy = []
for i in range(0, 10):
    X_train_init, X_validation_init, X_test_init, y_train_init, y_validation_init, y_test_init = prepare_datasets(old_data, 0.15, 0.15)


    X_train_new, X_validation_new, X_test_new, y_train_new, y_validation_new, y_test_new = prepare_datasets(new_data, 0.15, 0.15)

    # Create a new model by combining the pretrained model and the new output layer
    model = tf.keras.Sequential(pretrained_model.layers[:-1])
    model.add(tf.keras.layers.Dense(67, activation='softmax'))

    #Optionally, freeze the layers from the pretrained model
    # for layer in pretrained_model.layers[:-1]:
    #     layer.trainable = False

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.summary()

    # Start measuring time
    start_time = time.time()

    # train model
    history = model.fit(np.vstack([X_train_init, X_train_new]), np.vstack([y_train_init, y_train_new]), 
                        validation_data=(np.vstack([X_validation_init, X_validation_new]), np.vstack([y_validation_init, y_validation_new])),
                        batch_size=32, epochs=60)

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
    #plot_history(history, os.path.join(models_folder, 'train_history_mfcc_model_4l_60e_retrained_3'))

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(np.vstack([X_test_new, X_test_init]), np.vstack([y_test_new, y_test_init]), verbose=2)
    print('\nTest accuracy:', test_acc)
    accuracy.append(test_acc)

    # Save the model to the models folder
    #model.save(os.path.join(models_folder, 'mfcc_model_4l_60e_04d_retrained_3.h5'))

    #with open('training_history_mfcc_model_4l_60e_04d_retrained_3.json', 'w') as json_file:
    #    json.dump(history.history, json_file)

print('Avg accuracy:', np.mean(accuracy))