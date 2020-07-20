import pdb
import os
import os.path as osp
import time
import librosa
import h5py
import random
import math
import numpy as np
from glob import glob
import mmcv
import natsort
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from data.sep_dataset import SepDataset
from data.stereo_dataset import StereoDataset

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def process_image(image, augment):
    image = image.resize((480,240))
    w,h = image.size
    w_offset = w - 448
    h_offset = h - 224
    left = random.randrange(0, w_offset + 1)
    upper = random.randrange(0, h_offset + 1)
    image = image.crop((left, upper, left+448, upper+224))

    if augment:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

class SepStereoDataset(StereoDataset, SepDataset):
    def __init__(self, opt):
        self.opt = opt
        SepDataset.__init__(self, opt)
        StereoDataset.__init__(self, opt)

        random.seed(1234)
        dup_times = 5
        self.audios *= dup_times # in order to align with the length of MUSIC dataset
        random.shuffle(self.audios)
        random.shuffle(self.total_samples)

    def __getitem__(self, index):
        data_ret = self._get_stereo_item(index)
        data_ret.update(self._get_sep_item(index))

        return data_ret

    def __len__(self):
        return min(len(self.audios), len(self.total_samples))

    def name(self):
        return 'SepStereoDataset'
