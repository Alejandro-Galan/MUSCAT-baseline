## Musescore weights
pretrained_weights=weights/musescore/audio_ekern.ckpt

python3 -u process_audio_file.py --input_audio_folder test_audio --output_path_folder test_audio --krn_encoding ekern --checkpoint_path $pretrained_weights

