import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_3d_cnn(input_shape):
    model = Sequential()

    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001), padding='same'))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))

    model.add(Dense(units=2, activation='softmax'))  

    return model