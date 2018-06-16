"""
A set of useful functions for manipulating interactions
with the Sonic environment.
"""
import numpy as np
import skimage
from skimage.viewer import ImageViewer

def preprocess_obs(obs, size):
    """
    Transforms an input observation (an image from the sonic game)
    to the input dimensions and remove color.
    
    Observations from the sonic game are 320x224 pixel RGB images.
    with shape (224, 320, 3) - 224 rows, 320 columns, stack 3 times.
    """
    # 
    
    img = skimage.color.rgb2gray(obs)

    visualize_img(img)

    resized_img = skimage.transform.resize(img, size)
    visualize_img(resized_img)

def visualize_img(img):
    """
    Converts an input matrix to a visual image.
    Useful for training and observing the result of preprocessing.
    """
    viewer = ImageViewer(img)
    viewer.show()
    

