# List datasets into train/val/test sets. Removes bigger data
import os, torch
from data.encoding import krnParser
from data.preprocessing import preprocess_audio

DATASET="muscat/" #"musescore/"
root_folder = 'grandstaff/' + DATASET
partition_folder = 'grandstaff/partitions/' + DATASET
root_ekrns = os.path.join(root_folder, "ekrn")
root_wavs = os.path.join(root_folder, "wav")

FORCED_MAX_SEQ = 2056 #2540 #2056
FORCED_MAX_AUD = 2567

SOS_TOKEN = "<sos>"  # Start-of-sequence token
EOS_TOKEN = "<eos>"  # End-of-sequence token



KRN_PARSER = krnParser(encoding="ekern")

def preprocess_transcript(path: str):
    y = KRN_PARSER.encode(file_path=path)
    y = [SOS_TOKEN] + y + [EOS_TOKEN]
    return y


lens = {'Orig': [], 'Ref': []}

for file in os.listdir(partition_folder):
    path_set_file = os.path.join(partition_folder,file)
    # Read path_set_file by lines
    with open(path_set_file, 'r') as file:
        lines = file.readlines()
    new_lines = ""
    for line in lines:
        file_ekrn = os.path.join(root_ekrns, line.replace("\n", ".ekrn") )
        file_wav = os.path.join(root_wavs, line.replace("\n",'.wav') )

        if os.path.exists(file_ekrn) and os.path.exists(file_wav):
            
            x = preprocess_audio(path=file_wav)
            y = preprocess_transcript(path=file_ekrn)
            

            if len(y) > FORCED_MAX_SEQ or x.shape[2] > FORCED_MAX_AUD:
                print("AUD", x.shape, line)
                print("SEQ", len(y), line)
            else:
                new_lines += line #+ '\n'

    with open(path_set_file, 'w') as file:
        file.write(new_lines)

    lens['Orig'].append(len(lines))
    lens['Ref'].append(len(new_lines.split('\n')))
            
order = ["train", "test", "val"]    
for i, o in enumerate(order):
    print("For", o, "Original data", lens['Orig'][i], "refined data", lens['Ref'][i] )



