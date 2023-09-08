import json
import glob
import math
import mne
import numpy as np
import pathlib
import random
import tensorflow as tf

from eeg_utils import *
from env_handler import *

class y_index: # motor imagery
    LEFT_FIST = 0
    RIGHT_FIST = 1
    BOTH_FISTS = 2
    BOTH_FEET = 3

SINGLE_TASK = [4, 8, 12]
BOTH_TASK = [6, 10, 14]

mne.set_log_level(verbose='WARNING')

# Load in data
all_x = [] # each entry is a 2D array where rows are samples and columns are channels
all_y = [] # each entry is a 1D array [P(hold) P(reach)]

data_folder = pathlib.Path('C:/Users/maura/Downloads/eeg_db')
for subj_num in range(50, 61):
    for task_num in SINGLE_TASK + BOTH_TASK:
        edf = mne.io.read_raw_edf(data_folder / f'S{subj_num:03}' / f'S{subj_num:03}R{task_num:02}.edf')
        for anno in edf.annotations: # 'onset', 'duration', 'description'
            y = [0, 0, 0, 0]
            if anno['description'] == 'T0':
                continue
            elif anno['description'] == 'T1':
                if task_num in SINGLE_TASK:
                    y[y_index.LEFT_FIST] = 1
                else:
                    y[y_index.BOTH_FISTS] = 1
            elif anno['description'] == 'T2':
                if task_num in BOTH_TASK:
                    y[y_index.RIGHT_FIST] = 1
                else:
                    y[y_index.BOTH_FEET] = 1

            data = edf.copy().crop(tmin=anno['onset'], tmax=anno['onset'] + 3.99375, include_tmax=True).get_data(['C3..', 'C4..']).transpose()

            all_x.append(data)
            all_y.append(y)

assert len(all_x) == len(all_y)
num_items = len(all_x)

###
# Separate into training and validation
###

num_train = math.floor(num_items * 0.7) # 70/30
num_val = num_items - num_train
val_indices = random.sample(list(range(num_items)), num_val)
train_x, train_y, val_x, val_y = [], [], [], []
for i in range(num_items):
    if i in val_indices:
        val_x.append(all_x[i])
        val_y.append(all_y[i])
    else:
        train_x.append(all_x[i])
        train_y.append(all_y[i])

train_x = np.array(train_x)
val_x = np.array(val_x)
val_y = np.array(val_y)
train_y = np.array(train_y)

###
# Define the model
###
model_selection = 1

if model_selection == 2:
    input_shape = (640, 2, 1)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=25, kernel_size=(11, 1), input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    model.summary()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])

elif model_selection == 1: # CNN 5 TODO (Lun et al. 2020), HARD CODED 640 (5s of 128Hz, 4s of 160Hz)
    # Layer 1
    input_shape=(640,2,1)
    model = tf.keras.models.Sequential()

    # Layer 2
    model.add(tf.keras.layers.Conv2D(filters=25, kernel_size=(11, 1), input_shape=input_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    #model.add(tf.keras.layers.SpatialDropout2D(0.5)) # opt dropout

    # Layer 3
    model.add(tf.keras.layers.Conv2D(filters=25, kernel_size=(1, 2)))
    #model.add(tf.keras.layers.BatchNormalization()) # opt batch
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Layer 4
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1)))

    # Layer 5
    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(11, 1)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    #model.add(tf.keras.layers.SpatialDropout2D(0.5)) # opt dropout

    # Layer 6
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1)))

    # Layer 7
    model.add(tf.keras.layers.Conv2D(filters=100, kernel_size=(11, 1)))
    #model.add(tf.keras.layers.BatchNormalization()) # opt batch
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    #model.add(tf.keras.layers.SpatialDropout2D(0.5)) # opt dropout

    # Layer 8
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1)))

    # Layer 9
    model.add(tf.keras.layers.Conv2D(filters=200, kernel_size=(11, 1)))
    #model.add(tf.keras.layers.BatchNormalization()) # opt batch
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Layer 10
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))

    # Layer 11
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax))

    model.summary()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])

batch_size = 1
num_epoch = 2000

train_x = train_x[:2]
train_y = train_y[:2]
val_x = train_x[0:1]
val_y = train_y[0:1]

print(train_y)
quit()
model_log = model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(val_x, val_y))
score = model.evaluate(val_x, val_y, verbose=0)
print(f'Val loss: {score[0]}')
print(f'Val accuracy: {score[1]}')
model.save('output/eeg_model')
