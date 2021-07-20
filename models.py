import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

def nn_model(lr=0.01):

    model = Sequential()

    model.add(Dense(units=1000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=3, activation='softmax'))

    opt=tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model





