import tensorflow as tf
import os

from prepare_dataset import prepare_datasets

# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the models folder
models_folder = os.path.join(script_dir, '..', 'models')
# Load the pre-trained model
pretrained_model = tf.keras.models.load_model(os.path.join(models_folder, 'mfcc_model_4l_60e_04d_retrained.h5'))

new_data = "../data_new_users.json"

# get train, validation, test splits
print("Start loading data...")

X_train_new, X_validation_new, X_test_new, y_train_new, y_validation_new, y_test_new = prepare_datasets(new_data, 0.15, 0.15)

# evaluate model on test set
test_loss, test_acc = pretrained_model.evaluate(X_test_new, y_test_new, verbose=2)
print('\nTest accuracy:', test_acc)



