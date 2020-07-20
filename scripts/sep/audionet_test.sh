python demo_sep.py --input_audio_length 10 \
    --hop_size 0.1 \
    --weights_visual checkpoints/AudioNet_sep/visual_best.pth \
    --weights_audio checkpoints/AudioNet_sep/audio_best.pth \
    --output_dir_root eval_demo/sep/AudioNet_sep_best
