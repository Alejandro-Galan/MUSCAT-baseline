## Musescore weights
pretrained_weights=weights/musescore/audio_ekern.ckpt

python3 -u test.py --ds_name muscat --krn_encoding ekern --input_modality audio --checkpoint_path $pretrained_weights --use_distorted_images

