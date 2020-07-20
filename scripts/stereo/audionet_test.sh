python demo_stereo.py --input_audio_length 10 \
    --hop_size 0.1 \
    --weights_visual checkpoints/AudioNet/visual_best.pth \
    --weights_audio checkpoints/AudioNet/audio_best.pth \
    --output_dir_root eval_demo/stereo/AudioNet_best \
    --hdf5FolderPath YOUR_FAIR_PLAY_TEST_FILE_PATH 
