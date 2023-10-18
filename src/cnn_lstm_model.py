import tensorflow as tf

def build_model(input_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    print(input_shape)

    m = tf.keras.layers.Conv1D(32, 3, activation='relu')(input)
    m = tf.keras.layers.Conv1D(64, 3, activation='relu')(m)
    m = tf.keras.layers.MaxPooling1D()(m)
    m = tf.keras.layers.Dropout(0.25)(m)
    m = tf.keras.layers.Dense(128, activation='relu')(m)  
    m = tf.keras.layers.LSTM(128, return_sequences=True)(m)
    m = tf.keras.layers.LSTM(128, return_sequences=False)(m) 
    m = tf.keras.layers.Dense(256, activation='relu')(m) 
    m = tf.keras.layers.Flatten()(m)    
    m = tf.keras.layers.Dense(128, activation='relu')(m) 
    m = tf.keras.layers.Dense(61, activation='sigmoid')(m) 
        
    return tf.keras.Model(input, m) 
    