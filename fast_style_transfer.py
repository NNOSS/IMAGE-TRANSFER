from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K

from argparse import ArgumentParser
import numpy as np

img_x, img_y = 256, 256

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
    return img

# parse content and style input from terminal
parser = build_parser()
options = parser.parse_args()
content_path = options.content
model_path = options.model
output_path = options.output

# LOAD Model
content_image = K.variable(preprocess_image(content_path))
model = load_model(model_path)

# predict output
output = model.predict(content_image,verbose=1)
print(output.shape)
