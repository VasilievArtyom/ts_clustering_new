import math
import numpy as np
import pandas as pd
from numpy import *
from os import path
import os
import sys
import matplotlib.pyplot as plt

import keras
from keras.utils import Sequence
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, LSTM, InputLayer
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import losses
from keras.utils import plot_model
from tensorflow.python.client import device_lib


plt.rc('text', usetex=True)
np.random.seed(0)  # Set a random seed for reproducibility

# <--------------------->
# T U N A B L E
gpu_id = str(0)
WEIGHTSVERVION = 0
latent_dim = 3
# <--------------------->

if len(sys.argv) > 1:
    gpu_id = str(sys.argv[1])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# <--------------------->
# T U N A B L E
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
batch_size = 16000
# <--------------------->
print(device_lib.list_local_devices())


def read_clip_and_slice():
    feature_names = ['cmd',
                     'state0', 'state1', 'state2', 'state3', 'state4', 'state5', 
                     'data0', 'data1', 'data2', 'data3', 'data4', 'data5',
                     'is_rab', 'is_rez', 'is_rem', 'is_reg', 'is_ispr', 'is_neispr', 'is_otkaz', 'is_zavisim', 'is_vkl', 'is_vkl2',
                     'is_trig', 'is_trba', 'is_vedushciy', 'is_mu', 'is_blokirovka'
    ]
    clip_constants_invers = {'cmd': [2.],
         'state0': 256., 'state1': 256., 'state2': 256., 'state3': 256., 'state4': 256., 'state5': 256.,
         'data0': 256., 'data1': 256., 'data2': 256., 'data3': 256., 'data4': 256., 'data5': 256.
    }
    clip_constants = {'cmd': [1/2.],
         'state0': 1/256., 'state1': 1/256., 'state2': 1/256., 'state3': 1/256., 'state4': 1/256., 'state5': 1/256.,
         'data0': 1/256., 'data1': 1/256., 'data2': 1/256., 'data3': 1/256., 'data4': 1/256., 'data5': 1/256.
    }
    inpath = "../raw/"
    currentfile = path.join(inpath, "train_mssa_dataset.csv")
    df = pd.read_csv(currentfile)
    df.drop(columns=['Unnamed: 0', 'fpo_work_mode'], inplace=True)
    for row, clip_constant in clip_constants.items():
        df[row] *= clip_constant
    dataset_full = df[feature_names].to_numpy()
    np.random.shuffle(dataset_full)
    full_len = len(dataset_full[:, 0])
    train_len = int(full_len * 0.85)
    validation_len = full_len - train_len
    # train, test = dataset_full[:train_len, :], dataset_full[-validation_len:, :]
    return dataset_full[:train_len, :], dataset_full[-validation_len:, :]

train, test = read_clip_and_slice()
print("{} train samples".format(len(train)))
print("{} test samples".format(len(test)))


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = Sequential([
        	InputLayer((28, ), name='Encoder_input'),
            Dense(10, activation='relu', name='Encoder_hidden_large'),
            Dense(5, activation='relu', name='Encoder_hidden_middle'),
            Dense(3, activation='relu', name='Encoder_hidden_small'),
            Dense(latent_dim, activation='relu', name='Encoder_output')
        ])
        self.decoder = Sequential([
        	InputLayer((latent_dim, ), name='Decoder_input'),
            Dense(3, activation='sigmoid', name='Decoder_hidden_large'),
            Dense(5, activation='sigmoid', name='Decoder_hidden_middle'),
            Dense(10, activation='sigmoid', name='Decoder_hidden_small'),
            Dense(28, activation='sigmoid', name='Decoder_output'),
        ])
  
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# def index_generator(indexes):
#     while True:
#         for j in indexes:
#             yield j

# train_index = index_generator(indexes=range(len(train)))
# test_index = index_generator(indexes=range(len(test)))

# def batch_generator(batch_size, dataset, index_stream):
#     while True:
#         X1_shape = (batch_size, 28)
#         X1 = np.zeros(shape=X1_shape, dtype=np.float16)

#         for i in range(batch_size):
#             index = next(index_stream)
#             X1[i, :] = dataset[index, :]
#         yield X1, X1

# generator_traindata = batch_generator(batch_size=batch_size, dataset=train, index_stream=train_index)
# generator_validdata = batch_generator(batch_size=batch_size, dataset=test, index_stream=test_index)

autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss='mse')
path_checkpoint = 'weights/version{}.keras'.format(WEIGHTSVERVION)
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)
callbacks = [callback_early_stopping, callback_checkpoint, callback_reduce_lr]

try:
    autoencoder.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

steps_per_epoch = int(len(train)/ batch_size)
validation_steps = int(len(test) / batch_size)
print("steps_per_epoch = ", steps_per_epoch)
print("validation_steps = ", validation_steps)

history = autoencoder.fit(train, train, epochs=1000000, batch_size=batch_size,
                          # steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                          validation_data=[test, test], callbacks=callbacks)

# history = autoencoder.fit_generator(generator=generator_traindata, epochs=1000000,
#                                     steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
#                                     validation_data=generator_validdata, callbacks=callbacks)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(history.history["loss"], label="Training Loss")
ax.plot(history.history["val_loss"], label="Validation Loss")
plt.grid()
plt.ylabel(r'MSE')
plt.xlabel(r'Epoch')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig('weights/version{}.png'.format(WEIGHTSVERVION))
plt.clf()