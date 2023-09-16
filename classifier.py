import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import datasets, layers, models

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), padding='same', input_shape=(224, 224, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu")) 
    model.add(layers.Conv2D(8, (3, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))  
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(16, (3, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(16, (3, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1024,activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(1024,activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(7, activation='softmax'))
    #model.summary()
    STEPS_PER_EPOCH = 9356//32
    
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
          0.001,
          decay_steps=STEPS_PER_EPOCH*50,
          decay_rate=0.5,
          staircase=True)
    
    #lr_schedule = 0.001
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
    return model