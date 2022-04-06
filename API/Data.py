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
        trainDataset =  keras.preprocessing.image_dataset_from_directory(self.dataDirectory, seed = 123, 
                                                                         image_size = (self.datasetImageHeight, self.datasetImageWidth), 
                                                                         batch_size = self.batchSize)

        trainDataset = trainDataset.astype("float32") / 255.0
        print(trainDataset.class_names)

        
        
        return trainDataset



