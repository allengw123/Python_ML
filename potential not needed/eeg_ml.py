import json
import glob
import math
import numpy as np
import pathlib
import random
import tensorflow as tf

from eeg_utils import *
from env_handler import *

class y_index:
    P_HOLD = 0
    P_REACH = 1

patient_name = 'pro00087153_0040'
output_folder = pathlib.Path('output') / patient_name
with (output_folder / 'file_metadata.json').open('r') as fp:
    metadata = json.load(fp)
specified_electrodes = ['C3', 'C4']
sample_rate = metadata['eeg_sample_rate']
spec_sample_rate = metadata['spec_sample_rate']
num_seconds = metadata['seconds']

all_x = [] # each entry is a 2D array where rows are samples and columns are channels
all_y = [] # each entry is a 1D array [P(hold) P(reach)]

for file in glob.glob(str(output_folder / f'**/**/*_{specified_electrodes[0]}_eeg.csv')):
    _, _, trial, event, filename = pathlib.Path(file).parts
    y = [0, 0]

    event_id = filename.split('_')[0]
    if event == 'hold':
        y[y_index.P_HOLD] = 1
    elif event == 'reach':
        y[y_index.P_REACH] = 1
    else:
        continue

    all_y.append(y)

    x = np.zeros((num_seconds * spec_sample_rate, len(specified_electrodes)))
    for i, electrode in enumerate(specified_electrodes):
        x[:, i] = np.loadtxt(output_folder / trial / event / f'{event_id}_{electrode}_eeg.csv')
    all_x.append(x)

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

val_x = np.expand_dims(np.array(val_x), axis=-1)
val_y = np.array(val_y)
train_x = np.expand_dims(np.array(train_x), axis=-1)
train_y = np.array(train_y)

###
# Define the model
###
model_selection = 1

if model_selection == 1: # CNN 5 TODO (Lun et al. 2020), HARD CODED 640 (5s of 128Hz)
    input_shape=(1024,2,1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=25, kernel_size=(11, 1), input_shape=input_shape, activation=tf.keras.activations.relu))
    #model.add(tf.keras.layers.SpatialDropout2D(0.5)) # opt dropout
    model.add(tf.keras.layers.Conv2D(filters=25, kernel_size=(1, 2), activation=tf.keras.activations.relu))
    #model.add(tf.keras.layers.BatchNormalization()) # opt batch
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1)))
    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(11, 1), activation=tf.keras.activations.relu))
    #model.add(tf.keras.layers.SpatialDropout2D(0.5)) # opt dropout
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1)))
    model.add(tf.keras.layers.Conv2D(filters=100, kernel_size=(11, 1), activation=tf.keras.activations.relu))
    #model.add(tf.keras.layers.BatchNormalization()) # opt batch
    #model.add(tf.keras.layers.SpatialDropout2D(0.5)) # opt dropout
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1)))
    model.add(tf.keras.layers.Conv2D(filters=200, kernel_size=(11, 1), activation=tf.keras.activations.relu))
    #model.add(tf.keras.layers.BatchNormalization()) # opt dropout
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2))
    model.summary()
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

batch_size = 1
num_epoch = 100
model_log = model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(val_x, val_y))
score = model.evaluate(val_x, val_y, verbose=0)
print(f'Val loss: {score[0]}')
print(f'Val accuracy: {score[1]}')
model.save('output/eeg_model')
