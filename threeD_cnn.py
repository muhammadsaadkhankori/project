import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

def extract_frames_from_video(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(length // num_frames, 1)

    frame_count = 0
    saved_count = 0

    while cap.isOpened() and saved_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
            saved_count += 1
        frame_count += 1

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((112, 112, 3)))

    return frames

def load_videos_and_labels(dataset_dir):
    print("Loading dataset")
    classes = ['violence', 'non-violence']
    X = []
    y = []

    for class_name in classes:
        class_path = os.path.join(dataset_dir, class_name)
        label = 1 if class_name == 'violence' else 0

        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)
            frames = extract_frames_from_video(video_path)
            X.append(frames)
            y.append(label)

    return np.array(X), np.array(y)

def create_3d_cnn(input_shape):
    model = Sequential()

    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001), padding='same'))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
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
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(units=256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))

    model.add(Dense(units=2, activation='softmax'))  

    return model

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as 'training_history.png'")

def main():
    dataset_dir = 'D:/Muhammad_Saad/Ensemble-learning/dataset/'
    Videos, Labels = load_videos_and_labels(dataset_dir)
    
    train_features, test_features, train_labels, test_labels = train_test_split(Videos, Labels, test_size=0.05, random_state=42)
    train_labels = to_categorical(train_labels, num_classes=2)
    train_features = train_features / 255.0

    input_shape = (30, 112, 112, 3)
    optimizer = Adam(learning_rate=0.0001)
    batchSize = 32
    epochs = 2
    validation_split = 0.2

    model = create_3d_cnn(input_shape)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) 
    model.summary()

    history = model.fit(train_features, train_labels,
                        epochs=epochs,
                        batch_size=batchSize,
                        validation_split=validation_split
                        )

    plot_history(history)
    model.save("CNN_model.h5")
    print("Model saved as 'CNN_model.h5'")

if __name__ == "__main__":
    main()
