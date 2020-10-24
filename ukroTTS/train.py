import argparse

import torch.nn as nn
from torch.optim import Adam

from ukroTTS.models.deep_voice_3 import DeepVoice3, DeepVoice3Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-d', '--dataset', default=None, type=str,
                        help='path to the dataset dir (default: None)')
    parser.add_argument('-n', '--experiment_name', default=None, type=str,
                        help='path to the output dir (default: None)')
    parser.add_argument('-r', '--restore_epoch', default=-1, type=int,
                        help='restore epoch number (default: -1)')

    args = parser.parse_args()

    # init config
    config = DeepVoice3Config(args.config)

    # create model
    model = DeepVoice3(config)

    # optimizer and losses
    optimizer = Adam(model.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2), eps=config.eps,
                     weight_decay=config.weight_decay)

    mel_loss = nn.L1Loss()
    linear_loss = nn.L1Loss()
    done_loss = nn.BCELoss()

    datasets = {'train': LexiconDataset(args.dataset, split='train'),
                  'dev': LexiconDataset(args.dataset, split='dev')}

    trainer = Trainer(model, datasets, optimizer, loss, epochs=100, batch_size=256,
                      experiment_name=args.experiment_name,
                      logging_freq=10, restore_epoch=args.restore_epoch)
    trainer.train_and_validate()
