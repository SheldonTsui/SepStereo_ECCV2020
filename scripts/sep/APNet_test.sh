python demo_sep.py --input_audio_length 10 \
    --hop_size 0.1 \
    --fusion_model APNet \
    --use_fusion_pred \
    --weights_visual checkpoints/APNet_sep/visual_best.pth \
    --weights_audio checkpoints/APNet_sep/audio_best.pth \
    --weights_fusion checkpoints/APNet_sep/fusion_best.pth \
    --output_dir_root eval_demo/sep/APNet_sep_best
