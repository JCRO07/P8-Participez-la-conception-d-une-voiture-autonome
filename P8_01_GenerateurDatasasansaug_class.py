# +
import numpy as np

import cv2
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import albumentations as A
from skimage.transform import resize
from albumentations import (
    Compose, RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, OpticalDistortion, GaussNoise
)



class GenerateurDatasansaug(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, path_label, list_IDs, list_labels, n_classes, prob, batch_size=32, shuffle=False, preprocessing=None):
        'Initialization'
        self.batch_size = batch_size
        self.list_labels = list_labels
        self.path = path
        self.path_label = path_label
        self.list_IDs = list_IDs
        self.prob = prob
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.preprocessing = preprocessing
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(indexes)
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print('liste des images:')
        #print(list_IDs_temp)
        list_IDs_label_temp = [self.list_labels[k] for k in indexes]
        #print('liste des labels:')
        #print(list_IDs_label_temp)
        #print('valeur de prob:')
        #print(self.prob)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_IDs_label_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_IDs_label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #aug = A.Compose([
        #    A.HorizontalFlip(p=0.5),              
        #    A.OpticalDistortion(distort_limit=2, shift_limit=0.9, p=0.5),
        #    A.RandomContrast(limit=0.5, p=1)
        #    ]
        #)

        X = np.empty((self.batch_size,256, 256, 3))
        y = np.empty((self.batch_size, 256, 256))
        # Augmented data
        for i in range(len(list_IDs_temp)):
            # augment data
            image = cv2.imread(self.path + list_IDs_temp[i])
            image = tf.cast(image/255.0, tf.float32)            
            image = resize(image, (256, 256))
            mask = cv2.imread(self.path_label + list_IDs_label_temp[i])
            mask = tf.cast(mask/255.0, tf.float32)            
            mask = resize(mask, (256, 256))
            #augmented = aug(image=image, mask=mask)
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                #X, y = sample['image'], sample['mask']
                X[i,] = sample['image']
                # Store mask
                y[i,] = mask[...,0]
            else:
            # Store sample
                X[i,] = image

                # Store mask
                y[i,] = mask[...,0]
            
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
# -

