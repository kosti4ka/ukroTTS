import configparser
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils import model_zoo

from ukroTTS.models.modules import Encoder, Decoder, Converter

model_obj = namedtuple('model_obj', ['url', 'config_path'])

pretrained_models = {
    'ukro-deep-voice3-base': model_obj(
        url='',
        config_path=Path(__file__).parent / '../configs/ukro_deep_voice3_base.config'
    )
}


class DeepVoice3Config(dict):
    def __init__(self, model_config_file):
        super(DeepVoice3Config, self).__init__()

        # reading config file
        config_file = configparser.ConfigParser()
        config_file.read(model_config_file, encoding='utf8')

        # audio config
        self.n_fft = int(config_file['AudioConfig']['n_fft'])

        # model config
        self.max_spec_length = int(config_file['ModelConfig']['max_spec_length'])
        self.max_text_length = int(config_file['ModelConfig']['max_text_length'])
        self.reduction_factor = int(config_file['ModelConfig']['reduction_factor'])
        self.converter_conv_width = int(config_file['ModelConfig']['converter_conv_width'])
        self.dropout_prob = float(config_file['ModelConfig']['dropout_prob'])

        # encoder config
        self.n_tokens = int(config_file['EncoderConfig']['n_tokens'])
        self.encoder_n_hidden = int(config_file['EncoderConfig']['encoder_n_hidden'])
        self.encoder_conv_width = int(config_file['EncoderConfig']['encoder_conv_width'])
        self.encoder_conv_channels = int(config_file['EncoderConfig']['encoder_conv_channels'])
        self.encoder_position_rate = float(config_file['EncoderConfig']['encoder_position_rate'])

        # decoder config
        self.decoder_n_hidden = int(config_file['DecoderConfig']['decoder_n_hidden'])
        self.decoder_conv_width = int(config_file['DecoderConfig']['decoder_conv_width'])
        self.attention_n_hidden = int(config_file['DecoderConfig']['attention_n_hidden'])
        self.decoder_position_rate = float(config_file['DecoderConfig']['decoder_position_rate'])

        # optimizer config
        self.lr = float(config_file['OptimizerConfig']['lr'])
        self.beta_1 = float(config_file['OptimizerConfig']['beta_1'])
        self.beta_2 = float(config_file['OptimizerConfig']['beta_2'])
        self.eps = float(config_file['OptimizerConfig']['eps'])
        self.weight_decay = float(config_file['OptimizerConfig']['weight_decay'])


class PreTrainedDeepVoice3(nn.Module):
    def __init__(self, config):
        super(PreTrainedDeepVoice3, self).__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name):

        if model_name not in pretrained_models:
            raise ValueError

        # load config
        config = DeepVoice3Config(pretrained_models[model_name].config_path)  # TODO add metod from_file

        # instantiate model
        model = cls(config)

        # loading weights
        state_dict = model_zoo.load_url(pretrained_models[model_name].url,
                                        progress=True, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

        return model


class DeepVoice3(PreTrainedDeepVoice3):
    def __init__(self, config):
        super(DeepVoice3, self).__init__(config)

        # init
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.converter = Converter(config)


    def forward(self, x, y):
        # TODO rewrite desscription
        """

        Args:
            x: torch.LongTensor of character indices, shape (N,T)
            x_length: np.array of input length, shape (N,)
            y: torch.FloatTensor of mel-spectrograms, shape (N,T,80)

        Returns:
            torch.Tensor of linear scale spectrograms, shape (N,T,F)
        """

        keys, values = self.encoder(x)

        mel_out, decoder_out, done_out, alphas = self.decoder((keys, values), y)

        mel_out = mel_out.view(-1, 812, 80).contiguous()

        lin_out = self.converter(decoder_out)

        return mel_out, lin_out, done_out, alphas
