import joblib

import torch
import librosa
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

MEMORY = joblib.memory.Memory("./joblib_cache", mmap_mode="r", verbose=0)
NUM_CHANNELS = 1
AUDIO_HEIGHT = NUM_FREQ_BINS = 195
TOTENSOR = transforms.ToTensor()


def get_spectrogram_from_file(path: str) -> np.ndarray:
    y, fs = librosa.load(path, sr=22050)
    stft_fmax = 2093
    stft_frequency_filter_max = librosa.fft_frequencies(sr=fs, n_fft=2048) <= stft_fmax

    stft = librosa.stft(y, hop_length=512, win_length=2048, window="hann")
    stft = stft[stft_frequency_filter_max]

    stft_db = librosa.amplitude_to_db(np.abs(np.array(stft)), ref=np.max)
    log_stft = ((1.0 / 80.0) * stft_db) + 1.0

    return log_stft


@MEMORY.cache
def preprocess_audio(path: str) -> torch.Tensor:
    # Get spectrogram (already normalized)
    x = get_spectrogram_from_file(path)
    # Convert to PyTorch tensor
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x)  # [1, freq_bins == NUM_FREQ_BINS, time_frames]
    return x


@MEMORY.cache
def preprocess_image(path: str, img_height: int = None) -> torch.Tensor:
    x = Image.open(path).convert("L")  # Convert to grayscale
    if img_height is not None:
        new_width = int(
            img_height * x.size[0] / x.size[1]
        )  # Get width preserving aspect ratio
        x = x.resize((new_width, img_height))  # Resize
    x = TOTENSOR(x)  # Convert to tensor (normalizes to [0, 1])
    return x


def pad_batch_inputs(x, pad_value: float = 0.0):
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    max_height = max(x, key=lambda sample: sample.shape[1]).shape[1]
    x = torch.stack(
        [
            F.pad(
                i,
                pad=(
                    0,  # left
                    max_width - i.shape[2],  # right
                    0,  # top
                    max_height - i.shape[1],  # bottom
                ),
                value=pad_value,
            )
            for i in x
        ],
        dim=0,
    )
    return x


def pad_batch_transcripts(x, dtype=torch.int32):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = x.type(dtype=dtype)
    return x


def ar_batch_preparation_unimodal(batch, pad_value: float = 0.0):
    x, xl, y = zip(*batch)
    # Zero-pad inputs (images or audios) to maximum batch inputs shape
    x = pad_batch_inputs(x, pad_value=pad_value)
    xl = torch.tensor(xl, dtype=torch.int32)
    # Decoder input: transcript[:-1]
    y_in = [i[:-1] for i in y]
    y_in = pad_batch_transcripts(y_in, dtype=torch.int64)
    # Decoder target: transcript[1:]
    y_out = [i[1:] for i in y]
    y_out = pad_batch_transcripts(y_out, dtype=torch.int64)
    return x, xl, y_in, y_out


def ar_batch_preparation_image(batch):
    # Background for scores is white
    return ar_batch_preparation_unimodal(batch, pad_value=1.0)


def ar_batch_preparation_audio(batch):
    # Background for spectrograms is black
    return ar_batch_preparation_unimodal(batch)


def ar_batch_preparation_multimodal(batch):
    xi, xli, xa, xla, y = zip(*batch)
    # Zero-pad inputs (images and audios) to maximum batch inputs shape
    xi = pad_batch_inputs(xi, pad_value=1.0)
    xli = torch.tensor(xli, dtype=torch.int32)
    xa = pad_batch_inputs(xa)
    xla = torch.tensor(xla, dtype=torch.int32)
    # Decoder input: transcript[:-1]
    y_in = [i[:-1] for i in y]
    y_in = pad_batch_transcripts(y_in, dtype=torch.int64)
    # Decoder target: transcript[1:]
    y_out = [i[1:] for i in y]
    y_out = pad_batch_transcripts(y_out, dtype=torch.int64)
    return xi, xli, xa, xla, y_in, y_out
