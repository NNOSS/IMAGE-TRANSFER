from keras.models import load_model
from keras.preprocessing import image
from keras.applications import vgg19
from keras import backend as K

from argparse import ArgumentParser
import numpy as np
from scipy import ndimage
from PIL import Image

img_x, img_y = 256, 256
content_weight = 1
style_weight = 5
total_variation_weight = 1e-6

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--model',
            dest='model', help='path of traiend model',
            metavar='MODEL', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    return parser

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_x, img_y))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# custon loss function for reference
def custom_loss(content_img, combination_img):
    # compute the neural style loss
    # first we need to define 4 util functions

    input_tensor = K.concatenate([content_img,
                                  style_img,
                                  combination_img], axis=0)

    # build the VGG16 network with our 3 images as input
    # the model will be loaded with pre-trained ImageNet weights
    vgg = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet', include_top=False)
    print('Model loaded.')

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])

    def gram_matrix(x):
        assert K.ndim(x) == 3
        if K.image_data_format() == 'channels_first':
            features = K.batch_flatten(x)
        else:
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
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
        channels = 3
        size = img_x * img_y
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    # an auxiliary loss function
    # designed to maintain the "content" of the
    # base image in the generated image


    def content_loss(base, combination):
        return K.sum(K.square(combination - base))

    # the 3rd loss function, total variation loss,
    # designed to keep the generated image locally coherent


    def total_variation_loss(x):
        assert K.ndim(x) == 4
        if K.image_data_format() == 'channels_first':
            a = K.square(x[:, :, :img_x - 1, :img_y - 1] - x[:, :, 1:, :img_y - 1])
            b = K.square(x[:, :, :img_x - 1, :img_y - 1] - x[:, :, :img_x - 1, 1:])
        else:
            a = K.square(x[:, :img_x - 1, :img_y - 1, :] - x[:, 1:, :img_y - 1, :])
            b = K.square(x[:, :img_x - 1, :img_y - 1, :] - x[:, :img_x - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

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
    loss += total_variation_weight * total_variation_loss(combination_img)
    return loss

# parse content and style input from terminal
parser = build_parser()
options = parser.parse_args()
content_path = options.content
style_path = 'cuphead.jpg'
model_path = options.model
output_path = options.output

# LOAD Model
imList = []
input_img = Image.open("/home/yjiang/IMAGE-TRANSFER/"+ content_path))
input_img = np.asarray(input_img).transpose((2,0,1))
imList.append(input_img)
# content_image= content_image.reshape(content_image.shape[0], img_x, img_y, 3)
# content_image = K.variable(preprocess_image(content_path))
style_img = K.variable(preprocess_image(style_path))
model = load_model(model_path, custom_objects={'custom_loss':custom_loss})

# predict output
output = model.predict(imList[0]),verbose=1)
output_img = Image.fromarray(output[0].transpose((1,2,0)), 'RGB')
output_img.save(output_path)
