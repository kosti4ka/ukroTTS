import librosa
from scipy import signal
import numpy as np


SAMPLE_RATE = 16000
NUM_FREQ = 1025
NUM_MELS = 80
FRAME_SHIFT_MS = 12.5
FRAME_LENGTH_MS = 50

PREEMPHASIS = 0.97
MIN_LEVEL_DB = -100
REF_LEVEL_DB = 20


def load_wav(path):
    return librosa.core.load(path, sr=SAMPLE_RATE)[0]


def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - REF_LEVEL_DB
    return _normalize(S)


def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)


def preemphasis(x):
    return signal.lfilter([1, -PREEMPHASIS], [1], x)


_mel_basis = None


def _normalize(S):
    return np.clip((S - MIN_LEVEL_DB) / -MIN_LEVEL_DB, 0, 1)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    n_fft = (NUM_FREQ - 1) * 2
    hop_length = int(FRAME_SHIFT_MS / 1000 * SAMPLE_RATE)
    win_length = int(FRAME_LENGTH_MS / 1000 * SAMPLE_RATE)
    return n_fft, hop_length, win_length


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    n_fft = (NUM_FREQ - 1) * 2
    return librosa.filters.mel(SAMPLE_RATE, n_fft, n_mels=NUM_MELS)
