import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from ukro_g2p.tokenization import G2PTokenizer

from ukroTTS.utils import audio

DEFAULT_TOKINIZER_NAME = 'ukro-base-uncased'


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):

    # init paths
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    words_path = in_dir / 'words.txt'
    data_path = in_dir / 'data'
    out_hdf_path = out_dir / 'data.h5'

    idx2word = {x.split()[1]: x.split()[0] for x in open(words_path, 'r', encoding='utf-8').read().split('\n') if x}

    media_paths = glob.glob(f'{str(data_path)}/*/*.wav')

    tokenizer = G2PTokenizer.from_pretrained(DEFAULT_TOKINIZER_NAME)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    for wav_path in media_paths:
        wav_path = Path(wav_path)
        word_idx = wav_path.stem.split('_')[0]
        spk_idx = wav_path.stem.split('_')[1]
        word = idx2word[word_idx]
        futures.append(executor.submit(partial(_process_utterance, index, wav_path, word, spk_idx, tokenizer)))
        index += 1

    length = 0
    frames = 0
    max_input_length = 0
    max_output_length = 0
    with h5py.File(out_hdf_path, 'w') as hdf:
        for future in tqdm(futures):
            spectrogram, mel_spectrogram, n_frames, word, word_ids, spk_idx, index, wav_path = future.result()
            hdf_group = hdf.create_group(str(index))
            hdf_group['spectrogram'] = spectrogram
            hdf_group['mel_spectrogram'] = mel_spectrogram
            hdf_group['n_frames'] = n_frames
            hdf_group['word'] = word
            hdf_group['word_ids'] = word_ids
            hdf_group['spk_idx'] = spk_idx
            hdf_group['wav_name'] = wav_path.name

            length += 1
            frames += n_frames
            max_input_length = max(n_frames, max_input_length)
            max_output_length = max(len(word_ids), max_output_length)

    hours = frames * audio.FRAME_SHIFT_MS / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (length, frames, hours))
    print('Max input length:  %d' % max_input_length)
    print('Max output length: %d' % max_output_length)


def _process_utterance(index, wav_path, word, spk_idx, tokenizer):

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    # print(tokenizer.tokenize_graphemes(word))
    word_ids = np.asarray(tokenizer.convert_graphemes_to_ids(tokenizer.tokenize_graphemes(word)), dtype=np.int32)
    return spectrogram, mel_spectrogram, n_frames, word, word_ids, spk_idx, index, wav_path
