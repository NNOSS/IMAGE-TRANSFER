#!/usr/bin/env python
from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import keras
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.applications import vgg19
from keras.layers import (Input,
                          Conv2D,
                          Conv2DTranspose,
                          Activation,
                          BatchNormalization,
                          Add,
                          Lambda)
from keras.callbacks import TensorBoard

import tensorflow as tf

from argparse import ArgumentParser
from scipy import ndimage
import numpy as np
import time

batch_size = 4
epochs = 5
learning_rate = 1e-3


# input image dimensions
img_x, img_y = 256, 256

# define weight of content, style and total variation
content_weight = 1
style_weight = 5
total_variation_weight = 1e-6

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=False)
    parser.add_argument('--style',
            dest='style', help='style image',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=False)
    return parser

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_x, img_y))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def _conv_layer(x, num_filters, kernal_size, strides, padding='same', relu=True, input_shape=None):
    if input_shape == None:
        x = Conv2D(num_filters, kernel_size=kernal_size, strides=strides,
                   padding=padding)(x)
    else:
        x = Conv2D(num_filters, kernel_size=kernal_size, strides=strides,
                   padding=padding,
                   input_shape=input_shape)(x)
    x = BatchNormalization(axis=-1)(x)
    if(relu):
        x = Activation('relu')(x)
    return x

def _residual_block(x, filter_size=3):
    tmp = _conv_layer(x, 128, filter_size, 1)
    return Add()([x, _conv_layer(tmp, 128, filter_size, 1, relu=False)])

def _conv_transpose_layer(x, num_filters, kernal_size, strides, padding='same', relu=True):
    x = Conv2DTranspose(num_filters, kernel_size=kernal_size, strides=strides,
                        padding=padding)(x)
    x = BatchNormalization(axis=-1)(x)
    if (relu):
        x = Activation('relu')(x)
    return x

# parse content and style input from terminal
parser = build_parser()
options = parser.parse_args()
style_path = options.style
style_image = K.variable(preprocess_image(style_path))

# # this will contain our generated image
# if K.image_data_format() == 'channels_first':
#     combination_image = K.placeholder((1, 3, img_x, img_y))
# else:
#     combination_image = K.placeholder((1, img_x, img_y, 3))

class LossCalculator:
    def __init__(self, style_img):
        self.vgg19 = vgg19.VGG19(weights='imagenet', include_top=False)
        self.style_img = style_img

    def custom_loss(self, content_img, combination_img):
        # compute the neural style loss
        # first we need to define 4 util functions

        input_tensor = K.concatenate([content_img,
                                      self.style_img,
                                      combination_img], axis=0)

        # build the VGG16 network with our 3 images as input
        # the model will be loaded with pre-trained ImageNet weights
        vgg = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
        # # TODO: separate model definition from loss calculation
        # vgg = vgg19.VGG19(input_tensor=input_tensor,
        #                     weights='imagenet', include_top=False)
        print('Model loaded.')

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])

        def gram_matrix(x):
            assert K.ndim(x) == 3
            if K.image_data_format() == 'channels_first':
                features = K.batch_flatten(x)
            else:
                features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
            img_size = 256 * 256 * 3
            gram = K.dot(features, K.transpose(features)) / img_size
            print(gram)
            return gram

        # the "style loss" is designed to maintain
        # the style of the reference image in the generated image.
        # It is based on the gram matrices (which capture style) of
        # feature maps from the style reference image
        # and from the generated image
        def style_loss(style, combination):
            assert K.ndim(style) == 3
            assert K.ndim(combination) == 3
            S = gram_matrix(style)
            C = gram_matrix(combination)
            img_size = style.shape[0].value*style.shape[1].value*style.shape[2].value
            return 2 * K.sum(K.square(S - C)) / img_size

        # an auxiliary loss function
        # designed to maintain the "content" of the
        # base image in the generated image
        def content_loss(base, combination):
            img_size = base.shape[0].value*base.shape[1].value*base.shape[2].value
            return 2*K.sum(K.square(combination - base))/img_size

        # the 3rd loss function, total variation loss,
        # designed to keep the generated image locally coherent
        def total_variation_loss(x):
            assert K.ndim(x) == 4
            img_size = 1
            if K.image_data_format() == 'channels_first':
                a = K.square(x[:, :, :img_x - 1, :img_y - 1] - x[:, :, 1:, :img_y - 1])
                b = K.square(x[:, :, :img_x - 1, :img_y - 1] - x[:, :, :img_x - 1, 1:])
            else:
                a = K.square(x[:, :img_x - 1, :img_y - 1, :] - x[:, 1:, :img_y - 1, :])
                b = K.square(x[:, :img_x - 1, :img_y - 1, :] - x[:, :img_x - 1, 1:, :])
            return K.sum(K.pow(a + b, 1.25)) / img_size

        # combine these loss functions into a single scalar
        loss = K.variable(0.)
        layer_features = outputs_dict['block5_conv2']
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += content_weight * content_loss(base_image_features,
                                              combination_features)

        feature_layers = ['block1_conv1', 'block2_conv1',
                          'block3_conv1', 'block4_conv1',
                          'block5_conv1']
        for layer_name in feature_layers:
            layer_features = outputs_dict[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_features, combination_features)
            loss += (style_weight / len(feature_layers)) * sl
        # loss += total_variation_weight * total_variation_loss(combination_img)
        return loss


# load the microsoft COCO image dataset
imList = []
img_count = 0
total_count = len(os.listdir("training/train2014"))
print(total_count)
for imageName in sorted(os.listdir("training/train2014")):
    print(imageName)
    img = ndimage.imread("/home/nnoss/IMAGE-TRANSFER/training/train2014/" + imageName)
    if img.size==196608: # checking if img has 3-channel color, so 256*256*3 = 196608
        imList.append(ndimage.imread("/home/nnoss/IMAGE-TRANSFER/training/train2014/" + imageName, mode="RGB").transpose((2,0,1)))
    img_count += 1
    if img_count % (total_count / 1000) == 0:
        print("1% of image loaded")
        break
print('imList shape' + str(len(imList)))
img_train = np.asarray(imList, dtype="float32")
print('img_train shape1' + str(img_train.shape))

# define loss calculator
loss_calculator = LossCalculator(style_image)

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, RGB colour images would have 3
img_train = img_train.reshape(img_train.shape[0], img_x, img_y, 3)
print('img_train shape2' + str(img_train.shape))
input_shape = (img_x, img_y, 3)

# convert the data to the right type
img_train = img_train.astype('float32')
img_train /= 255.
print('x_train shape3:', img_train.shape)
print(img_train.shape[0], 'train samples')

input1 = Input(shape=input_shape);
# Convolutional Layers
conv1 = _conv_layer(input1, num_filters=32, kernal_size=(9,9), strides=(1,1), input_shape=input_shape)
conv2 = _conv_layer(conv1, num_filters=64, kernal_size=(3,3), strides=(2,2))
conv3 = _conv_layer(conv2, num_filters=128, kernal_size=(3,3), strides=(2,2))

#Residual blocks
# res1 = _residual_block(conv3, 3)
# res2 = _residual_block(res1, 3)
# res3 = _residual_block(res2, 3)
# res4 = _residual_block(res3, 3)
# res5 = _residual_block(res4, 3)


# Conv2DTranspose / Deconvolutional layers
# deconv1 = _conv_transpose_layer(res5, num_filters=64, kernal_size=(3,3), strides=(2,2))
deconv1 = _conv_transpose_layer(conv3, num_filters=128, kernal_size=(3,3), strides=(2,2))
deconv2 = _conv_transpose_layer(deconv1, num_filters=64, kernal_size=(3,3), strides=(2,2))
deconv3 = _conv_transpose_layer(deconv2, num_filters=32, kernal_size=(9,9), strides=(1,1))

# deconv1 = _conv_transpose_layer(conv3, num_filters=64, kernal_size=(3,3), strides=(2,2))
# deconv2 = _conv_transpose_layer(deconv1, num_filters=32, kernal_size=(3,3), strides=(2,2))
# deconv3 = _conv_layer(deconv2, num_filters=3, kernal_size=(9,9), strides=(1,1), padding="same", relu=False)
# deconv3 = _conv_layer(deconv2, num_filters=3, kernal_size=(9,9), strides=(1,1), padding="same", relu=False)
# pred = Activation('tanh')(deconv3)
# output = Add()([pred, input1])
# output = Activation('tanh')(output)
# # output = K.map_fn(lambda x: x*127.5+255./2, output)
# output = Lambda(lambda x: x*127.5 + 255./2)(output)
output = deconv3
# Train
model = Model(inputs=input1, outputs=output)
model.compile(loss=loss_calculator.custom_loss,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="./logs", histogram_freq=1, write_graph=True, write_images=True)
tensorboard.set_model(model)
history = model.fit(x=img_train, y=img_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[tensorboard], validation_data=([img_train, img_train])).history
model.save('transfer_model_convdeconv2.h5')
