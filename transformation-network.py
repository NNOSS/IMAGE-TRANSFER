#!/usr/bin/env python

from __future__ import print_function
import keras
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.application import vgg19
from keras.layers import (Input,
                          Conv2D,
                          Conv2DTranspose,
                          Activation,
                          BatchNormalization,
                          Add)

from argparse import ArgumentParser
import numpy as np

batch_size = 4
epochs = 5

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--style',
            dest='style', help='style image',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    return parser

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def gram_matrix():



def content_loss():
    content_model = Model(inputs=vgg.input, outputs=vgg.get_layer(CONTENT_LAYER).output)


def style_loss():



def total_variation_loss():



def _conv_layer(x, num_filters, kernal_size, strides, padding=None, relu=True):
    x = Conv2D(num_filters, kernel_size=kernal_size, strides=strides,
               padding=padding,
               input_shape=input_shape)(x)
    x = BatchNormalization(axis=1)(x)
    if(relu):
        x = Activation('relu')(x)
    return x

def _residual_block(x, num_filters=3):
    tmp = _conv_layer(x, 128, 1)
    return Add([x, _conv_layer(tmp, 128, num_filters, 1, relu=False)])

def _conv_transpose_layer(x, num_filters, kernal_size, strides, padding=None, relu=True):
    x = Conv2DTranspose(num_filters, kernel_size=kernal_size, strides=strides,
                        padding=padding,
                        input_shape=input_shape)(x)
    x = BatchNormalization(axis=1)(x)
    return x


# parse conten and style input from terminal
parser = build_parser()
options = parser.parse_args()
content_path = options.content
style_path = options.style
output_path = options.output

content_img = preprocess_image(content_path)
style_img = preprocess_image(style_path)
content = image.img_to_array(content)

CONTENT_LAYERS = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

vgg = vgg19.VGG19(weights='imagenet', include_top=False)

# input image dimensions
img_x, img_y = 256, 256

# TODO: load the microsoft COCO image dataset


# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, RGB colour images would have 3
img_train = img_train.reshape(img_train.shape[0], img_x, img_y, 3)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
img_train = img_train.astype('float32')
img_train /= 255
print('x_train shape:', img_train.shape)
print(img_train.shape[0], 'train samples')

input1 = Input(shape=input_shape);
# Convolutional Layers
conv1 = _conv_layer(input1, num_filters=32, kernal_size=(9,9), strides=(1,1), padding="same")
conv2 = _conv_layer(conv1, num_filters=64, kernal_size=(3,3), strides=(2,2))
conv3 = _conv_layer(conv2, num_filters=128, kernal_size=(3,3), strides=(2,2))

#Residual blocks
res1 = _residual_block(conv3, 3)
res2 = _residual_block(res1, 3)
res3 = _residual_block(res2, 3)
res4 = _residual_block(res3, 3)
res5 = _residual_block(res4, 3)

# Conv2DTranspose / Deconvolutional layers
deconv1 = _conv_transpose_layer(res5, num_filters=64, kernal_size=(3,3), strides=(2,2))
deconv2 = _conv_transpose_layer(deconv1, num_filters=32, kernal_size=(3,3), strides=(2,2))
deconv3 = _conv_transpose_layer(deconv2, num_filters=3, kernal_size=(9,9), strides=(1,1), relu=False)
output = Activation('tanh')(deconv3)

# VGG net + loss function?

model = Model(inputs=img_train, outputs=output)
model.compile(loss=todo, # we need to define our own loss function
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])
