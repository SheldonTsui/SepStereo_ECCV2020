import os
import os.path as osp
import sys
import pdb
import argparse
import librosa
import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image
import subprocess
from options.test_options import TestOptions
import torchvision.transforms as transforms
import torch
import torchvision
from data.audioVisual_dataset import generate_spectrogram
from models.networks import VisualNet, VisualNetDilated, AudioNet, AssoConv, APNet, weights_init 

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return rms / desired_rms, samples

def main():
    #load test arguments
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")

    ## build network
    # visual net
    original_resnet = torchvision.models.resnet18(pretrained=True)
    if opt.visual_model == 'VisualNet':
        net_visual = VisualNet(original_resnet)
    elif opt.visual_model == 'VisualNetDilated':
        net_visual = VisualNetDilated(original_resnet)
    else:
        raise TypeError("please input correct visual model type")

    if len(opt.weights_visual) > 0:
        print('Loading weights for visual stream')
        net_visual.load_state_dict(torch.load(opt.weights_visual), strict=True)

    # audio net
    net_audio = AudioNet(
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
    )
    net_audio.apply(weights_init)
    if len(opt.weights_audio) > 0:
        print('Loading weights for audio stream')
        net_audio.load_state_dict(torch.load(opt.weights_audio), strict=True)

    # fusion net
    if opt.fusion_model == 'none':
        net_fusion = None
    elif opt.fusion_model == 'AssoConv':
        net_fusion = AssoConv()
    elif opt.fusion_model == 'APNet':
        net_fusion = APNet()
    else:
        raise TypeError("Please input correct fusion model type") 

    if net_fusion is not None and len(opt.weights_fusion) > 0:
        net_fusion.load_state_dict(torch.load(opt.weights_fusion), strict=True)

    net_visual.to(opt.device)
    net_audio.to(opt.device)
    net_visual.eval()
    net_audio.eval()
    if net_fusion is not None:
        net_fusion.to(opt.device)
        net_fusion.eval()

    test_h5_path = opt.hdf5FolderPath
    print("---Testing---: ", test_h5_path)
    testf = h5py.File(test_h5_path, 'r')
    audio_list = testf['audio'][:]

    # ensure output dir
    if not osp.exists(opt.output_dir_root):
        os.mkdir(opt.output_dir_root)

    for audio_file in tqdm(audio_list):
        audio_file = bytes.decode(audio_file)
        video_path = audio_file.replace('audio_resave', 'frames')[:-4]
        input_audio_path = audio_file
        video_frame_path = video_path
        audio_id = audio_file.split('/')[-1][:-4]
        cur_output_dir_root = os.path.join(opt.output_dir_root, audio_id)

        #load the audio to perform separation
        audio, audio_rate = librosa.load(input_audio_path, sr=opt.audio_sampling_rate, mono=False)
        audio_channel1 = audio[0,:]
        audio_channel2 = audio[1,:]

	#define the transformation to perform on visual frames
        vision_transform_list = [transforms.Resize((224,448)), transforms.ToTensor()]
        vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        vision_transform = transforms.Compose(vision_transform_list)

	#perform spatialization over the whole audio using a sliding window approach
        overlap_count = np.zeros((audio.shape)) #count the number of times a data point is calculated
        binaural_audio = np.zeros((audio.shape))

	#perform spatialization over the whole spectrogram in a siliding-window fashion
        sliding_window_start = 0
        data = {}
        samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
        while sliding_window_start + samples_per_window < audio.shape[-1]:
            sliding_window_end = sliding_window_start + samples_per_window
            normalizer, audio_segment = audio_normalize(audio[:,sliding_window_start:sliding_window_end])
            audio_segment_channel1 = audio_segment[0,:]
            audio_segment_channel2 = audio_segment[1,:]
            audio_segment_mix = audio_segment_channel1 + audio_segment_channel2

            audio_diff = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
            audio_mix = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 + audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
            #get the frame index for current window
            frame_index = int(round((((sliding_window_start + samples_per_window / 2.0) / audio.shape[-1]) * opt.input_audio_length + 0.05) * 10 ))
            image = Image.open(os.path.join(video_frame_path, str(frame_index) + '.jpg')).convert('RGB')
            #image = image.transpose(Image.FLIP_LEFT_RIGHT)
            frame = vision_transform(image).unsqueeze(0) #unsqueeze to add a batch dimension
            # data to device
            audio_diff = audio_diff.to(opt.device)
            audio_mix = audio_mix.to(opt.device)
            frame = frame.to(opt.device)

            vfeat = net_visual(frame) 
            if net_fusion is not None: 
                upfeatures, output = net_audio(audio_diff, audio_mix, vfeat, return_upfeatures=True)
                output.append(net_fusion(audio_mix, vfeat, upfeatures)) 
            else:
                output = net_audio(audio_diff, audio_mix, vfeat)

	    #ISTFT to convert back to audio
            if opt.use_fusion_pred:
                pred_left_spec = output['pred_left'][0,:,:,:].data[:].cpu().numpy()
                pred_left_spec = pred_left_spec[0,:,:] + 1j * pred_left_spec[1,:,:]
                reconstructed_signal_left = librosa.istft(pred_left_spec, hop_length=160, win_length=400, center=True, length=samples_per_window)
                pred_right_spec = output['pred_right'][0,:,:,:].data[:].cpu().numpy()
                pred_right_spec = pred_right_spec[0,:,:] + 1j * pred_right_spec[1,:,:]
                reconstructed_signal_right = librosa.istft(pred_right_spec, hop_length=160, win_length=400, center=True, length=samples_per_window)
            else:
                predicted_spectrogram = output['binaural_spectrogram'][0,:,:,:].data[:].cpu().numpy()
                reconstructed_stft_diff = predicted_spectrogram[0,:,:] + (1j * predicted_spectrogram[1,:,:])
                reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=160, win_length=400, center=True, length=samples_per_window)
                reconstructed_signal_left = (audio_segment_mix + reconstructed_signal_diff) / 2
                reconstructed_signal_right = (audio_segment_mix - reconstructed_signal_diff) / 2
            reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) * normalizer

            binaural_audio[:,sliding_window_start:sliding_window_end] = binaural_audio[:,sliding_window_start:sliding_window_end] + reconstructed_binaural
            overlap_count[:,sliding_window_start:sliding_window_end] = overlap_count[:,sliding_window_start:sliding_window_end] + 1
            sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)

	#deal with the last segment
        normalizer, audio_segment = audio_normalize(audio[:,-samples_per_window:])
        audio_segment_channel1 = audio_segment[0,:]
        audio_segment_channel2 = audio_segment[1,:]
        audio_diff = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
        audio_mix = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 + audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
	#get the frame index for last window
        frame_index = int(round(((opt.input_audio_length - opt.audio_length / 2.0) + 0.05) * 10))
        image = Image.open(os.path.join(video_frame_path, str(frame_index) + '.jpg')).convert('RGB')
	#image = image.transpose(Image.FLIP_LEFT_RIGHT)
        frame = vision_transform(image).unsqueeze(0) #unsqueeze to add a batch dimension
        # data to device
        audio_diff = audio_diff.to(opt.device)
        audio_mix = audio_mix.to(opt.device)
        frame = frame.to(opt.device)

        vfeat = net_visual(frame) 
        if net_fusion is not None: 
            upfeatures, output = net_audio(audio_diff, audio_mix, vfeat, return_upfeatures=True)
            output.update(net_fusion(audio_mix, vfeat, upfeatures)) 
        else:
            output = net_audio(audio_diff, audio_mix, vfeat)

	#ISTFT to convert back to audio
        if opt.use_fusion_pred:
            pred_left_spec = output['pred_left'][0,:,:,:].data[:].cpu().numpy()
            pred_left_spec = pred_left_spec[0,:,:] + 1j * pred_left_spec[1,:,:]
            reconstructed_signal_left = librosa.istft(pred_left_spec, hop_length=160, win_length=400, center=True, length=samples_per_window)
            pred_right_spec = output['pred_right'][0,:,:,:].data[:].cpu().numpy()
            pred_right_spec = pred_right_spec[0,:,:] + 1j * pred_right_spec[1,:,:]
            reconstructed_signal_right = librosa.istft(pred_right_spec, hop_length=160, win_length=400, center=True, length=samples_per_window)
        else:
            predicted_spectrogram = output['binaural_spectrogram'][0,:,:,:].data[:].cpu().numpy()
            reconstructed_stft_diff = predicted_spectrogram[0,:,:] + (1j * predicted_spectrogram[1,:,:])
            reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=160, win_length=400, center=True, length=samples_per_window)
            reconstructed_signal_left = (audio_segment_mix + reconstructed_signal_diff) / 2
            reconstructed_signal_right = (audio_segment_mix - reconstructed_signal_diff) / 2
        reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) * normalizer

        #add the spatialized audio to reconstructed_binaural
        binaural_audio[:,-samples_per_window:] = binaural_audio[:,-samples_per_window:] + reconstructed_binaural
        overlap_count[:,-samples_per_window:] = overlap_count[:,-samples_per_window:] + 1

	#divide aggregated predicted audio by their corresponding counts
        predicted_binaural_audio = np.divide(binaural_audio, overlap_count)

	#check output directory
        if not os.path.isdir(cur_output_dir_root):
            os.mkdir(cur_output_dir_root)

        mixed_mono = (audio_channel1 + audio_channel2) / 2
        librosa.output.write_wav(os.path.join(cur_output_dir_root, 'predicted_binaural.wav'), predicted_binaural_audio, opt.audio_sampling_rate)
        librosa.output.write_wav(os.path.join(cur_output_dir_root, 'mixed_mono.wav'), mixed_mono, opt.audio_sampling_rate)
        librosa.output.write_wav(os.path.join(cur_output_dir_root, 'input_binaural.wav'), audio, opt.audio_sampling_rate)

if __name__ == '__main__':
    main()
