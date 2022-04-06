import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import PIL
import time
from tqdm import tqdm
from tensorflow import keras
from keras import Model
from keras.optimizers import adam_v2
from keras.layers import Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, LeakyReLU, Reshape, Rescaling

class GAN:
    
    def __init__(self, latent_dim):

        self.latent_dim = latent_dim
        self.channels = 3

    def Generator(self):

        gen_input = Input(shape=self.latent_dim,)

            # If shape error occurs, adjust values here
        x = Dense(64 * 8 * 8)(gen_input)
        x = LeakyReLU()(x)
        x = Reshape((8, 8, 64))(x)
        
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        
        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        
        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        
        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        
        x = Conv2D(512, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(self.channels, 7, activation='tanh', padding='same')(x)
        # print(x.shape)
        
        generator = Model(gen_input, x)
        return generator

    def Discriminator(self):

        dis_input = Input(shape=(64, 64, self.channels))
  
        x = Conv2D(256, 3)(dis_input)
        x = LeakyReLU()(x)
    
        x = Conv2D(256, 4, strides=2)(x)
        x = LeakyReLU()(x)
        
        x = Conv2D(256, 4, strides=2)(x)
        x = LeakyReLU()(x)
        
        x = Conv2D(256, 4, strides=2)(x)
        x = LeakyReLU()(x)
        
        x = Conv2D(256, 4, strides=2)(x)
        x = LeakyReLU()(x)
        
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(1, activation='sigmoid')(x)
        discriminator = Model(dis_input, x)
        
        optimizer = keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5)
        
        discriminator.compile(
            optimizer=optimizer,
            loss='binary_crossentropy'
        )
        
        return discriminator