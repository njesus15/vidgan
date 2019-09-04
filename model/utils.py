import os

from torch import nn
from torchvision import transforms, utils
from model.model import VideoGen
from keras.preprocessing import image
import torch
from skimage import img_as_ubyte
import imageio
import numpy as np


def get_model(filename=None):
    """
    :return: Video Generator Model

    """
    models = []

    if not isinstance(filename, list):
        filename = [filename]

    if len(filename) > 1:
        for model_path in filename:
            generator = VideoGen()
            generator.load_state_dict(torch.load(model_path, map_location='cpu'))
            generator.eval()
            models.append(generator)
    else:
        model_path = filename[0]
        generator = VideoGen()
        generator.load_state_dict(torch.load(model_path, map_location='cpu'))
        generator.eval()
        models = generator

    return models

def model_output(model, filename):

    assert os.path.exists(filename)
    img = image.load_img(filename, target_size=(64, 64))
    small_img = img
    trans = transforms.ToTensor()
    img = trans(img)
    norm_t = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    video = norm_t(img.float())
    img = video.unsqueeze(0)

    return small_img, model(img)

def make_gif(images, filename):
    # Receives [3,32,64,64] tensor, and saves in gif format

    if not isinstance(images, np.ndarray):
        x = images.permute(1, 2, 3, 0)
        x = x.numpy()
    else:
        x = images
    x = img_as_ubyte(x)
    frames = []
    for i in range(32):
        frames += [x[i]]
    imageio.mimsave(filename, frames, duration=0.1)

def denorm(image):
    # denormalize tensor image
    out = (image + 1.0) / 2.0
    tf = nn.Tanh()
    return tf(out)





