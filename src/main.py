import os
import keras

from dataset_load import load_dataset
from save_dataset_to_json import save_dataset_to_json
from load_dataset_from_json import load_dataset_from_json
from prepare_dataset import prepare_datasets
from cnn_basic_model import build_model
from plot_history import plot_history
 
# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the dataset directory
dataset_dir = os.path.join(script_dir, '..', 'data')

json_filename = "../data.json"

# need to execute only on the first run

#mfcc, labels = load_dataset(dataset_dir)
#save_dataset_to_json(mfcc, labels, json_filename)


# get train, validation, test splits
print("Start loading data...")
X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(json_filename, 0.15, 0.15)
# print(X_train.shape)
# print(X_test.shape)
# print(X_validation.shape)
# print(y_train.shape)
# print(y_train[0])

input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = build_model(input_shape)

# compile model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

model.summary()

# train model
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100)

# plot accuracy/error for training and validation
plot_history(history)

# evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
