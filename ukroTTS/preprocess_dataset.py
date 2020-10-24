import argparse
from multiprocessing import cpu_count
from pathlib import Path

from tqdm import tqdm

from ukroTTS.datasets import ukro_words


def preprocess_ukro_words(args):
    in_dir = Path(args.base_dir)
    out_dir = Path(args.base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ukro_words.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='data/ukro_words_dataset')
    parser.add_argument('--output', default='hdfs')
    parser.add_argument('--dataset', choices=['ukro_words'], default='ukro_words')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()
    if args.dataset == 'ukro_words':
        preprocess_ukro_words(args)
    else:
        raise NotImplementedError
