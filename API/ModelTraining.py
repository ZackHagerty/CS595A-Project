
from Data import Dataset
from GAN.Model import GAN
import tensorflow as tf
import numpy as np
import os
import PIL
import time
from tensorflow import keras
from keras import Model
from keras.optimizers import adam_v2
from keras.layers import Input

class AnimeDragan:

    def __init__(self):

        self.latent_dim = 32
        self.channels = 3
        self.width = 64
        self.height = 64

        models = GAN(self.latent_dim)
        self.generator = models.Generator()
        self.discriminator = models.Discriminator()
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim, ))
        gan_output = self.discriminator(self.generator(gan_input))

        # GAN Model
        gan = Model(gan_input, gan_output)
        optimizer = keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5)
        gan.compile(optimizer=optimizer, loss='binary_crossentropy')

    def training(self, images):
        iters = 25000
        batch_size = 16
        res_dir = './content/waifus'
        file_path = '%s/generated_%d.png'
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)
        control_size_sqrt = 6
        control_vectors = np.random.normal(size=(control_size_sqrt**2, self.latent_dim)) / 2
        start = 0
        d_losses = []
        a_losses = []
        images_saved = 0

        for step in range(iters):
            start_time = time.time()
            latent_vectors = np.random.normal(size=(batch_size, self.latent_dim))
            generated = self.generator.predict(latent_vectors)
            
            real = images[start:start + batch_size]
            combined_images = np.concatenate([generated, real])
            
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            labels += .05 * np.random.random(labels.shape)
            
            d_loss = self.discriminator.train_on_batch(combined_images, labels)
            d_losses.append(d_loss)
            
            latent_vectors = np.random.normal(size=(batch_size, self.latent_dim))
            misleading_targets = np.zeros((batch_size, 1))
            
            a_loss = self.gan.train_on_batch(latent_vectors, misleading_targets)
            a_losses.append(a_loss)
            
            start += batch_size
            if start > images.shape[0] - batch_size:
                start = 0
            
            if step % 50 == 49:
                self.gan.save_weights('gan.h5')
                
                print('%d/%d: d_loss: %.4f,  a_loss: %.4f.' % (step + 1, iters, d_loss, a_loss))
                
                control_image = np.zeros((self.width * control_size_sqrt, self.height * control_size_sqrt, self.channels))
                control_generated = self.generator.predict(control_vectors)
                for i in range(control_size_sqrt ** 2):
                    x_off = i % control_size_sqrt
                    y_off = i // control_size_sqrt
                    control_image[x_off * self.width:(x_off + 1) * self.width, y_off * self.height:(y_off + 1) * self.height, :] = control_generated[i, :, :, :]
                im = PIL.Image.fromarray(np.uint8(control_image * 255))
                im.save(file_path % (res_dir, images_saved))
                images_saved += 1



if __name__ == "__main__":
    
   # modelcreation = AnimeDragan()
    dataClass = Dataset("./archive/animefaces256cleaner", datasetImageHeight=64, datasetImageWidth=64)

    dataset = dataClass.createDataset()
 
   # modelcreation.training(dataset)
