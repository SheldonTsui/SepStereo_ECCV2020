import pdb
import os
import os.path as osp
import time
import librosa
import random
import math
import numpy as np
from glob import glob
import mmcv
import natsort
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms

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

class SepDataset(Dataset):
    def __init__(self, opt):
        Dataset.__init__(self)

        self.opt = opt

        # load mono audio list
        self.audio_margin = 0.1
        audio_list_file = os.path.join(opt.MUSICPath, opt.mode+".csv")
        self.total_samples = mmcv.list_from_file(audio_list_file) 

        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        random.seed(1234)

        self.exp_audio_len = int(self.opt.audio_length * self.opt.audio_sampling_rate)

    def _get_sep_item(self, index):
        # we can only support 2 mono samples simultaneously
        N = 2
        chosen_samples = [self.total_samples[index]]
        if self.opt.mode != 'train':
            random.seed(index)
        # avoid repeat sample 
        for i in range(1, N):
            while True:
                new_sample = random.choice(self.total_samples)
                if new_sample not in chosen_samples:
                    chosen_samples.append(new_sample)
                    break
 
        audio_sep_list = []
        frame_sep_list = []
        for idx, chosen_sample in enumerate(chosen_samples):
            audio_file, img_folder = chosen_sample.split(',')
            #load audio
            audio_sep, audio_rate = librosa.load(audio_file, sr=self.opt.audio_sampling_rate, mono=True)
            #randomly get a start time for the audio segment from the 10s clip
            audio_len = len(audio_sep) / audio_rate
            assert audio_len - self.opt.audio_length - self.audio_margin > self.audio_margin
            audio_start_time = random.uniform(self.audio_margin, audio_len - self.opt.audio_length - self.audio_margin)
            audio_end_time = audio_start_time + self.opt.audio_length
            audio_start = int(audio_start_time * self.opt.audio_sampling_rate)
            audio_end = audio_start + self.exp_audio_len
            audio_sep = normalize(audio_sep[audio_start:audio_end])
            audio_sep_list.append(audio_sep)

            #load frame
            cur_img_list = natsort.natsorted(glob(osp.join(img_folder, '*.jpg')))
            # get the closest frame to the audio segment
            frame_idx_sep = (audio_start_time + audio_end_time) / 2 * 10
            frame_idx_sep = int(np.clip(frame_idx_sep, 0, len(cur_img_list) - 1))
            frame_sep = process_image(Image.open(cur_img_list[frame_idx_sep]).convert('RGB'), self.opt.enable_data_augmentation)
            frame_sep = self.vision_transform(frame_sep)
            frame_sep_list.append(frame_sep)

        left_channel, right_channel = audio_sep_list
        #passing the spectrogram of the difference
        sep_diff_spec = torch.FloatTensor(generate_spectrogram(left_channel - right_channel))
        sep_mix_spec = torch.FloatTensor(generate_spectrogram(left_channel + right_channel))

        if self.opt.mode == 'train':
            return {'frame_sep': frame_sep_list, 'sep_diff_spec':sep_diff_spec, 'sep_mix_spec':sep_mix_spec}
        else:
            return {'frame_sep': frame_sep_list, 'sep_diff_spec':sep_diff_spec, 'sep_mix_spec':sep_mix_spec,
                    'left_audio': left_channel, 'right_audio': right_channel}

    def __getitem__(self, index):
        return self._get_sep_item(index)

    def __len__(self):
        return len(self.total_samples)

    def name(self):
        return 'SepDataset'
