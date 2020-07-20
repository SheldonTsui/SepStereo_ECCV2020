python demo_sep.py --input_audio_length 10 \
    --hop_size 0.1 \
    --fusion_model AssoConv \
    --use_fusion_pred \
    --weights_visual checkpoints/AssoConv_sep/visual_best.pth \
    --weights_audio checkpoints/AssoConv_sep/audio_best.pth \
    --output_dir_root eval_demo/sep/AssoConv_sep_best
