import os
import os.path as osp
import sys
import pdb
import argparse
import librosa
import natsort
import numpy as np
import mmcv
import random
from mir_eval.separation import bss_eval_sources
from glob import glob
from tqdm import tqdm
import h5py
from PIL import Image
import subprocess
from options.test_options import TestOptions
import torchvision.transforms as transforms
import torch
import torchvision
from data.sep_dataset import generate_spectrogram
from models.networks import VisualNet, VisualNetDilated, AudioNet, AssoConv, APNet, weights_init, Rearrange

def audio_empty(wav):
    flag = np.sum(np.abs(wav)) < 1e-3

    return flag 

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return rms / desired_rms, samples

def separation_metrics(pred_left, pred_right, gt_left, gt_right, mix):
    if audio_empty(gt_left) or audio_empty(gt_right) or audio_empty(pred_right) or audio_empty(pred_left) or audio_empty(mix):
        print("----------- Empty -----------")
        return None
    sdr, sir, sar, _ = bss_eval_sources(np.asarray([gt_left, gt_right]), np.asarray([pred_left, pred_right]), False)
    sdr_mix, _, _, _ = bss_eval_sources(np.asarray([gt_left, gt_right]), np.asarray([mix, mix]), False)

    return sdr.mean(), sir.mean(), sar.mean(), sdr_mix.mean() 

def main():
    #load test arguments
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")

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
        print('Loading weights for fusion stream')
        net_fusion.load_state_dict(torch.load(opt.weights_fusion), strict=True)

    net_visual.to(opt.device)
    net_audio.to(opt.device)
    net_visual.eval()
    net_audio.eval()
    if net_fusion is not None:
        net_fusion.to(opt.device)
        net_fusion.eval()

    # rearrange module
    net_rearrange = Rearrange()
    net_rearrange.to(opt.device)
    net_rearrange.eval()

    val_list_file = 'data/dummy_MUSIC_split/val.csv'
    sample_list = mmcv.list_from_file(val_list_file)

    # ensure output dir
    if not osp.exists(opt.output_dir_root):
        os.mkdir(opt.output_dir_root)

    #define the transformation to perform on visual frames
    vision_transform_list = [transforms.Resize((224,448)), transforms.ToTensor()]
    vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    vision_transform = transforms.Compose(vision_transform_list)
    chosen_audio_len = opt.audio_sampling_rate * 6
    total_metrics = {'sdr':[], 'sir':[], 'sar':[], 'sdr_m':[]}

    for global_idx, sample in enumerate(sample_list):
        N = 2
        chosen_samples = [sample]
        # avoid repeat sample
        for i in range(1, N):
            while True:
                new_sample = random.choice(sample_list)
                if new_sample not in chosen_samples:
                    chosen_samples.append(new_sample)
                    break
        audio_margin = 6
        audio_list = []
        frame_idx_list = []
        frame_list = []
        
        cur_output_dir_root = []
        for idx, chosen_sample in enumerate(chosen_samples):
            input_audio_path, img_folder, _, cate = chosen_sample.split(',') 
            cur_output_dir_root.append('_'.join([cate, img_folder[-4:]]))
            #load the audio to perform separation
            audio, audio_rate = librosa.load(input_audio_path, sr=opt.audio_sampling_rate, mono=True)
            #randomly get a start time for 6s audio segment
            audio_len = len(audio) / audio_rate
            audio_start_time = random.uniform(audio_margin, audio_len - 6 - audio_margin)
            audio_end_time = audio_start_time + 6
            audio_start = int(audio_start_time * opt.audio_sampling_rate)
            audio_end = audio_start + chosen_audio_len 
            audio = audio[audio_start:audio_end]
            audio_list.append(audio)
            
            #lock the frame idx range
            frame_list.append(natsort.natsorted(glob(osp.join(img_folder, '*.jpg'))))
            frame_idx_list.append(int((audio_start_time + audio_end_time) / 2 * 10))
        
	#perform spatialization over the whole audio using a sliding window approach
        overlap_count = np.zeros(chosen_audio_len) #count the number of times a data point is calculated
        pred_left = np.zeros(chosen_audio_len)
        pred_right = np.zeros(chosen_audio_len)

	#perform spatialization over the whole spectrogram in a siliding-window fashion
        sliding_window_start = 0
        sliding_idx = 0
        data = {}
        samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
        while sliding_window_start + samples_per_window < chosen_audio_len:
            sliding_window_end = sliding_window_start + samples_per_window
            normalizer1, audio_segment1 = audio_normalize(audio_list[0][sliding_window_start:sliding_window_end])
            normalizer2, audio_segment2 = audio_normalize(audio_list[1][sliding_window_start:sliding_window_end])
            audio_segment_channel1 = audio_segment1
            audio_segment_channel2 = audio_segment2
            audio_segment_mix = audio_segment_channel1 + audio_segment_channel2

            audio_diff = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
            audio_mix = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 + audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
            #get the frame index for current window
            frame_index1 = int(np.clip(frame_idx_list[0] + sliding_idx, 0, len(frame_list[0]) - 1))
            frame_index2 = int(np.clip(frame_idx_list[1] + sliding_idx, 0, len(frame_list[1]) - 1))
            image1 = Image.open(frame_list[0][frame_index1]).convert('RGB')
            image2 = Image.open(frame_list[1][frame_index2]).convert('RGB')
            #image = image.transpose(Image.FLIP_LEFT_RIGHT)
            frame1 = vision_transform(image1).unsqueeze(0).to(opt.device) #unsqueeze to add a batch dimension
            frame2 = vision_transform(image2).unsqueeze(0).to(opt.device) #unsqueeze to add a batch dimension
            # data to device
            audio_diff = audio_diff.to(opt.device)
            audio_mix = audio_mix.to(opt.device)

            img_feat = net_rearrange(net_visual(frame1), net_visual(frame2))
            if net_fusion is not None:
                upfeatures, output = net_audio(audio_diff, audio_mix, img_feat, return_upfeatures=True)
                output.update(net_fusion(audio_mix, img_feat, upfeatures))
            else:
                output = net_audio(audio_diff, audio_mix, img_feat, return_upfeatures=False)


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
            pred_left[sliding_window_start:sliding_window_end] = pred_left[sliding_window_start:sliding_window_end] + reconstructed_signal_left * normalizer1
            pred_right[sliding_window_start:sliding_window_end] = pred_right[sliding_window_start:sliding_window_end] + reconstructed_signal_right * normalizer2

            overlap_count[sliding_window_start:sliding_window_end] = overlap_count[sliding_window_start:sliding_window_end] + 1
            sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)
            sliding_idx += 1

	#deal with the last segment
        normalizer1, audio_segment1 = audio_normalize(audio_list[0][-samples_per_window:])
        normalizer2, audio_segment2 = audio_normalize(audio_list[1][-samples_per_window:])
        audio_segment_channel1 = audio_segment1
        audio_segment_channel2 = audio_segment2
        audio_diff = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
        audio_mix = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 + audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
	#get the frame index for last window
        frame_index1 = int(np.clip(frame_idx_list[0] + sliding_idx, 0, len(frame_list[0]) - 1))
        frame_index2 = int(np.clip(frame_idx_list[1] + sliding_idx, 0, len(frame_list[1]) - 1))
        image1 = Image.open(frame_list[0][frame_index1]).convert('RGB')
        image2 = Image.open(frame_list[1][frame_index2]).convert('RGB')
        #image = image.transpose(Image.FLIP_LEFT_RIGHT)
        frame1 = vision_transform(image1).unsqueeze(0).to(opt.device) #unsqueeze to add a batch dimension
        frame2 = vision_transform(image2).unsqueeze(0).to(opt.device) #unsqueeze to add a batch dimension
        # data to device
        audio_diff = audio_diff.to(opt.device)
        audio_mix = audio_mix.to(opt.device)

        img_feat = net_rearrange(net_visual(frame1), net_visual(frame2))
        if net_fusion is not None:
            upfeatures, output = net_audio(audio_diff, audio_mix, img_feat, return_upfeatures=True)
            output.update(net_fusion(audio_mix, img_feat, upfeatures))
        else:
            output = net_audio(audio_diff, audio_mix, img_feat, return_upfeatures=False)

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

        pred_left[-samples_per_window:] = pred_left[-samples_per_window:] + reconstructed_signal_left * normalizer1
        pred_right[-samples_per_window:] = pred_right[-samples_per_window:] + reconstructed_signal_right * normalizer2

        #add the spatialized audio to reconstructed_binaural
        overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1

	#divide aggregated predicted audio by their corresponding counts
        pred_left = np.divide(pred_left, overlap_count)
        pred_right = np.divide(pred_right, overlap_count)
        gt_left, gt_right = audio_list
        mix_audio = (gt_left + gt_right) / 2

        sep_results = separation_metrics(pred_left, pred_right, gt_left, gt_right, mix_audio)
        if sep_results is not None and global_idx % 20 == 0:
            sdr, sir, sar, sdr_m = sep_results
            print("index: {}, sdr: {}, sir: {}, sar: {}, sdr_m: {}\n".format(global_idx, sdr, sir, sar, sdr_m))
            total_metrics['sdr'].append(sdr)
            total_metrics['sir'].append(sir)
            total_metrics['sar'].append(sar)
            total_metrics['sdr_m'].append(sdr_m) 

	#check output directory
        cur_output_dir_root = osp.join(opt.output_dir_root, '+'.join(cur_output_dir_root))
        if not os.path.isdir(cur_output_dir_root):
            os.mkdir(cur_output_dir_root)

        librosa.output.write_wav(osp.join(cur_output_dir_root, 'pred_left.wav'), pred_left, sr=opt.audio_sampling_rate)
        librosa.output.write_wav(osp.join(cur_output_dir_root, 'pred_right.wav'), pred_right, sr=opt.audio_sampling_rate)
        librosa.output.write_wav(osp.join(cur_output_dir_root, 'gt_left.wav'), gt_left, sr=opt.audio_sampling_rate)
        librosa.output.write_wav(osp.join(cur_output_dir_root, 'gt_right.wav'), gt_right, sr=opt.audio_sampling_rate)
        librosa.output.write_wav(osp.join(cur_output_dir_root, 'mix.wav'), mix_audio, sr=opt.audio_sampling_rate)

    print_content = "----- sdr: {}, sir: {}, sar: {}, sdr_m: {} -----\n".format(
            sum(total_metrics['sdr']) / len(total_metrics['sdr']),
            sum(total_metrics['sir']) / len(total_metrics['sir']),
            sum(total_metrics['sar']) / len(total_metrics['sar']),
            sum(total_metrics['sdr_m']) / len(total_metrics['sdr_m'])
    )
    print(print_content)

if __name__ == '__main__':
    random.seed(1234)
    torch.manual_seed(1234)
    main()
