import os
import keras
import time
import json

from dataset_load import load_dataset
from save_dataset_to_json import save_dataset_to_json
from load_dataset_from_json import load_dataset_from_json
from prepare_dataset import prepare_datasets
from cnn_basic_model import build_model
from plot_history import plot_history
from create_spec_model_with_dataset import create_spectrogram_model_with_dataset
#from cnn_lstm_model import build_model

def create_basic_mfcc_model_with_dataset():
    json_filename = "../data.json"

    # need to execute only on the first run

    #mfcc, labels = load_dataset(dataset_dir)
    #save_dataset_to_json(mfcc, labels, json_filename)

    # get train, validation, test splits
    print("Start loading data...")
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(json_filename, 0.15, 0.15)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)
    return model, X_train, X_validation, X_test, y_train, y_validation, y_test


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the dataset directory
dataset_dir = os.path.join(script_dir, '..', 'data')

model, X_train, X_validation, X_test, y_train, y_validation, y_test = create_spectrogram_model_with_dataset()

# compile model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

model.summary()

# Start measuring time
start_time = time.time()

# train model
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=60)

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
plot_history(history, os.path.join(models_folder, 'train_history_spectrogram_image_4l_60e'))

# evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model to the models folder
model.save(os.path.join(models_folder, 'spectrogram_image_4l_60e.h5'))

with open('training_history_spectrogram_4l_60e.json', 'w') as json_file:
    json.dump(history.history, json_file)


