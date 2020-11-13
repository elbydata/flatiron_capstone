import numpy as np
from keras import layers
from keras import models
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import get_custom_objects


def add_swish():

    class Swish(layers.Activation):   
        def __init__(self, activation, **kwargs):
            super(Swish, self).__init__(activation, **kwargs)
            self.__name__ = 'swish'

    def swish(x):
        return (K.sigmoid(x)*x)

    get_custom_objects().update({'swish': Swish(swish)})