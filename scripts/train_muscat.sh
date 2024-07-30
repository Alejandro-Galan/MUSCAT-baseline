## Musescore weights
pretrained_weights=weights/musescore/audio_ekern.ckpt

# No finetuning
# python3 train.py muscat --krn_encoding ekern --input_modality audio --batch_size 1 --attn_window 100


# Finetuning
python3 train.py muscat --krn_encoding ekern --input_modality audio --batch_size 1 --attn_window 100 --checkpoint_path $pretrained_weights

