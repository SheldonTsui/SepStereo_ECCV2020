python demo_stereo.py --input_audio_length 10 \
    --hop_size 0.1 \
    --fusion_model APNet \
    --use_fusion_pred \
    --weights_visual checkpoints/APNet/visual_best.pth \
    --weights_audio checkpoints/APNet/audio_best.pth \
    --weights_fusion checkpoints/APNet/fusion_best.pth \
    --output_dir_root eval_demo/stereo/APNet_best \
    --hdf5FolderPath YOUR_FAIR_PLAY_TEST_FILE_PATH
