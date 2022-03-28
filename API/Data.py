import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras


class Dataset:

    def __init__(self, dataDirectory=None, batchSize = 32, datasetImageHeight=256, datasetImageWidth=256 ):

        self.dataDirectory = dataDirectory
        self.batchSize = batchSize
        self.datasetImageHeight = datasetImageHeight
        self.datasetImageWidth = datasetImageWidth


    def createDataset(self):
        dataset =  keras.preprocessing.image_dataset_from_directory(self.dataDirectory, label_mode = None, seed = 123,
                                                                         image_size = (self.datasetImageHeight, self.datasetImageWidth), 
                                                                         batch_size = self.batchSize)

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        trainDataset = dataset.map(lambda x, y: (normalization_layer(x), y))

        imageShape = (self.datasetImageHeight, self.datasetImageWidth)
        datasetLength = len(trainDataset)

        return trainDataset, imageShape, datasetLength



