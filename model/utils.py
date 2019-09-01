import imageio as imageio
import torch as torch
import os

from torch import nn
from torchvision import transforms, utils
from model.model import VideoGen
from keras.preprocessing import image
import torch
from skimage import img_as_ubyte
import imageio



def get_model():
    """
    :return: Video Generator Model

    """
    PATH = 'model/pytorch/test_gen_15fps.pt'
    generator = VideoGen()
    generator.load_state_dict(torch.load(PATH, map_location='cpu'))
    generator.eval()

    return generator

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

    x = images.permute(1, 2, 3, 0)
    x = x.numpy()
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





