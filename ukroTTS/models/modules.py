import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout_prob=0.05):
        super().__init__()
        self.dropout_prob = dropout_prob
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.utils.weight_norm(nn.Conv1d(channels, channels * 2,
                                                   kernel_size, stride=1,
                                                   dilation=dilation,
                                                   padding=padding),
                                         dim=None)

    def forward(self, x):
        residual = x
        x = F.dropout(x, self.dropout_prob)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return (x + residual) * math.sqrt(0.5)


class CausalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout_prob=0.05):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.input_buffer = None
        self.conv = nn.utils.weight_norm(nn.Conv1d(channels, channels * 2,
                                                   kernel_size, stride=1,
                                                   dilation=dilation,
                                                   padding=self.padding),
                                         dim=None)

    def forward(self, x):
        residual = x
        x = F.dropout(x, self.dropout_prob)
        x = self.conv(x)[:, :, :-self.padding]
        x = F.glu(x, dim=1)
        return (x + residual) * math.sqrt(0.5)

    def inference(self, x):
        residual = x
        x = F.dropout(x, self.dropout_prob)

        bsz = x.size(0)  # input: bsz x len x dim
        if self.kernel_size > 1:
            input = x.data
            if self.input_buffer is None:
                self.input_buffer = input.new(bsz, input.size(1), self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :, :-1] = self.input_buffer[:, :, 1:].clone()
            # append next input
            self.input_buffer[:, :, -1] = input[:, :, -1]
            input = torch.autograd.Variable(self.input_buffer, volatile=True)
            # if self.dilation > 1:
            #     input = input[:, :, 0::self.dilation].contiguous()
        x = self.conv(input)[:, :, self.padding:-self.padding]
        x = F.glu(x, dim=1)
        return (x + residual) * math.sqrt(0.5)


class AttentionLayer(nn.Module):
    def __init__(self, config, window_size=3):  # conv_channels, embed_dim, dropout=0.1):
        super(AttentionLayer, self).__init__()
        decoder_n_hidden = config.decoder_n_hidden
        encoder_n_hidden = config.encoder_n_hidden
        attention_n_hidden = config.attention_n_hidden
        self.dropout_prob = config.dropout_prob
        max_spec_length = config.max_spec_length
        reduction_factor = config.reduction_factor
        decoder_position_rate = config.decoder_position_rate
        max_text_length = config.max_text_length
        encoder_position_rate = config.encoder_position_rate
        self.window_size = window_size

        # Initialize in_projection_query and in_projection_keys with the same weigths
        self.in_projection_query = nn.Linear(decoder_n_hidden, attention_n_hidden)
        self.in_projection_keys = nn.Linear(encoder_n_hidden, attention_n_hidden)
        weight_init = torch.rand(attention_n_hidden, decoder_n_hidden).uniform_(-1. / math.sqrt(attention_n_hidden),
                                                                                1. / math.sqrt(attention_n_hidden))
        bias_init = torch.rand(attention_n_hidden).uniform_(-1. / math.sqrt(attention_n_hidden),
                                                            1. / math.sqrt(attention_n_hidden))
        with torch.no_grad():
            self.in_projection_query.weight.set_(weight_init)
            self.in_projection_query.bias.set_(bias_init)
            self.in_projection_keys.weight.set_(weight_init)
            self.in_projection_keys.bias.set_(bias_init)
        self.in_projection_query = nn.utils.weight_norm(self.in_projection_query, dim=None)
        self.in_projection_keys = nn.utils.weight_norm(self.in_projection_keys, dim=None)
        self.in_projection_values = nn.utils.weight_norm(nn.Linear(encoder_n_hidden, attention_n_hidden), dim=None)
        self.out_projection = nn.utils.weight_norm(nn.Linear(attention_n_hidden, decoder_n_hidden), dim=None)


        self.query_position_encoding = self.positional_encoding(max_spec_length // reduction_factor, decoder_n_hidden,
                                                                position_rate=decoder_position_rate)
        self.keys_position_encoding = self.positional_encoding(max_text_length, encoder_n_hidden,
                                                           position_rate=encoder_position_rate)

    def forward(self, query, keys, values, prev_max_attention_idx, position_encoding=False):
        residual = query

        if position_encoding:
            query = query + self.query_position_encoding
            keys = keys + self.keys_position_encoding

        # attention
        x = self.in_projection_query(query)
        keys = self.in_projection_keys(keys)
        values = self.in_projection_values(values)

        x = torch.bmm(x, keys.transpose(1, 2))

        mask_value = -float("inf")
        if prev_max_attention_idx is not None:
            if prev_max_attention_idx > 0:
                x[:, :, :prev_max_attention_idx] = mask_value
            ahead = prev_max_attention_idx + self.window_size
            if ahead < x.size(-1):
                x[:, :, ahead:] = mask_value

        # softmax over last dim
        # (B, tgt_len, src_len)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)

        attn_scores = x
        x = F.dropout(x, p=self.dropout_prob)
        x = torch.bmm(x, values)
        # scale attention output
        s = values.size(1)
        x = x * (s * math.sqrt(1.0 / s))  # TODO double check on it
        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def positional_encoding(self, num_timestamps, num_channels, position_rate=1.):

        # TODO do it in torch
        position_enc = np.array([[i * position_rate / np.power(10000, k / num_channels) for k in range(num_channels)]
                                 for i in range(num_timestamps)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        # TODO deal with cuda and no cuda cases
        return Variable(torch.from_numpy(position_enc).float())


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        n_tokens = config.n_tokens
        encoder_n_hidden = config.encoder_n_hidden
        dropout_prob = config.dropout_prob
        encoder_conv_width = config.encoder_conv_width
        encoder_conv_channels = config.encoder_conv_channels

        # # Text input embeddings
        # self.embed_tokens = nn.Embedding(n_vocab, embed_dim)
        #
        # # Non-causual convolutions
        # # in_channels = convolutions[0][0]
        # self.fc1 = nn.Linear(256, 128)
        #
        # kernel_size = 5
        # dilation = 1
        # pad = (kernel_size - 1) // 2 * dilation
        # dropout = 0.05

        self.embeding = nn.Embedding(n_tokens, encoder_n_hidden)
        self.in_projection = nn.utils.weight_norm(nn.Linear(encoder_n_hidden, encoder_conv_channels), dim=None)
        # TODO add loop instead of list
        self.encoder_conv_blocks = nn.Sequential(
            ConvBlock(encoder_conv_channels, kernel_size=encoder_conv_width, dropout_prob=dropout_prob),
            ConvBlock(encoder_conv_channels, kernel_size=encoder_conv_width, dropout_prob=dropout_prob),
            ConvBlock(encoder_conv_channels, kernel_size=encoder_conv_width, dropout_prob=dropout_prob),
            ConvBlock(encoder_conv_channels, kernel_size=encoder_conv_width, dropout_prob=dropout_prob),
            ConvBlock(encoder_conv_channels, kernel_size=encoder_conv_width, dropout_prob=dropout_prob),
            ConvBlock(encoder_conv_channels, kernel_size=encoder_conv_width, dropout_prob=dropout_prob),
            ConvBlock(encoder_conv_channels, kernel_size=encoder_conv_width, dropout_prob=dropout_prob)
        )
        self.out_projection = nn.utils.weight_norm(nn.Linear(encoder_conv_channels, encoder_n_hidden), dim=None)

    def forward(self, text_sequences):

        # embed text_sequences
        x = self.embeding(text_sequences)

        # x = F.dropout(x, p=0.05)
        residual = x

        # project to size of convolution
        x = self.in_projection(x)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # １D conv blocks
        x = self.encoder_conv_blocks(x)

        # B x C x T -> B x T x C
        x = x.transpose(1, 2)

        # project back to size of embedding
        keys = self.out_projection(x)

        values = (keys + residual) * math.sqrt(0.5)

        return keys, values


class Decoder(nn.Module):
    def __init__(self, config,
                 in_dim=80, r=4,
                 max_positions=512, padding_idx=None,
                 convolutions=((128, 5, 1),) * 4,
                 attention=True, dropout=0.05,
                 use_memory_mask=False,
                 force_monotonic_attention=True,
                 query_position_rate=1.0,
                 key_position_rate=1.29
                 ):
        super(Decoder, self).__init__()
        self.dropout_prob = config.dropout_prob
        decoder_n_hidden = config.decoder_n_hidden
        decoder_conv_width = config.decoder_conv_width
        self.in_dim = in_dim
        self.r = r

        in_channels = in_dim * r

        self.fc1 = nn.Linear(in_channels, 256)
        in_channels = convolutions[0][0]

        # # Causual convolutions
        # self.projections = nn.ModuleList()
        # self.convolutions = nn.ModuleList()
        # self.attention = nn.ModuleList()

        # Conv1dLayer = Conv1d if has_dilation(convolutions) else LinearizedConv1d

        # self.convolution_1 = nn.Conv1d(256, 256 * 2, kernel_size=5,
        #                     padding=(5 - 1) * 1, dilation=1)
        #
        # self.convolution_2 = nn.Conv1d(256, 256 * 2, kernel_size=5,
        #                                  padding=(5 - 1) * 2, dilation=2)

        self.convolution_1 = CausalConvBlock(decoder_n_hidden, kernel_size=decoder_conv_width,
                                             dropout_prob=self.dropout_prob, dilation=1)
        self.convolution_2 = CausalConvBlock(decoder_n_hidden, kernel_size=decoder_conv_width,
                                             dropout_prob=self.dropout_prob, dilation=2)
        self.convolution_3 = CausalConvBlock(decoder_n_hidden, kernel_size=decoder_conv_width,
                                             dropout_prob=self.dropout_prob, dilation=2)
        self.convolution_4 = CausalConvBlock(decoder_n_hidden, kernel_size=decoder_conv_width,
                                             dropout_prob=self.dropout_prob, dilation=3)

        self.attention_1 = AttentionLayer(config)
        # self.attention_2 = AttentionLayer(opt)
        # self.attention_3 = AttentionLayer(opt)
        self.attention_4 = AttentionLayer(config)

        self.fc2 = nn.Linear(256, in_dim * r)

        # # decoder states -> Done binary flag
        self.fc3 = nn.Linear(decoder_n_hidden, 1)

        # self._is_inference_incremental = False
        # self.max_decoder_steps = 200
        # self.min_decoder_steps = 10
        # self.use_memory_mask = use_memory_mask
        # if isinstance(force_monotonic_attention, bool):
        #     self.force_monotonic_attention = \
        #         [force_monotonic_attention] * len(convolutions)
        # else:
        #     self.force_monotonic_attention = force_monotonic_attention

    def forward(self, encoder_out, inputs=None,
                text_positions=None, frame_positions=None,
                speaker_embed=None, lengths=None, prev_max_attention_idx=None):

        # Grouping multiple frames if necessary
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(inputs.size(0), inputs.size(1) // self.r, -1)
        assert inputs.size(-1) == self.in_dim * self.r

        keys, values = encoder_out


        x = inputs
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # project to size of convolution
        x = F.relu(self.fc1(x), inplace=True)

        alignments = []

        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)
        x = self.convolution_1(x)

        # Feed conv output to attention layer as query
        # (B x T x C)
        x = x.transpose(1, 2)
        x, alignment = self.attention_1(x, keys, values, prev_max_attention_idx)
        alignments += [alignment]

        # (B x C x T)
        x = x.transpose(1, 2)
        x = self.convolution_2(x)
        # x = x.transpose(1, 2)
        # x, alignment = self.attention_2(x, keys, values)
        # alignments += [alignment]

        # (B x C x T)
        # x = x.transpose(1, 2)
        x = self.convolution_3(x)
        # x = x.transpose(1, 2)
        # x, alignment = self.attention_3(x, keys, values)
        # alignments += [alignment]

        # (B x C x T)
        # x = x.transpose(1, 2)
        x = self.convolution_4(x)
        x = x.transpose(1, 2)
        x, alignment = self.attention_4(x, keys, values, prev_max_attention_idx)
        alignments += [alignment]

        # well, I'm not sure this is really necesasary
        decoder_states = x


        # TODO take a look at korean and paper it was made in a differetn way there
        # project to mel-spectorgram
        x = F.sigmoid(self.fc2(decoder_states))

        # Done flag
        done = F.sigmoid(self.fc3(decoder_states))

        return x, decoder_states, done, alignments

    def inference(self, encoder_out):
        keys, values = encoder_out
        B = keys.size(0)


        decoder_states = []
        outputs = []
        alignments = []
        dones = []
        prev_max_attention_idx = [0] * 2  # since we have two attention layers


        t = 0
        initial_input = Variable(keys.data.new(B, 1, self.in_dim * self.r).zero_())
        current_input = initial_input

        while True:

            if t > 0:
                current_input = outputs[-1]

            x = current_input
            x = F.dropout(x, p=self.dropout_prob, training=self.training)

            # project to size of convolution
            x = F.relu(self.fc1(x), inplace=True)

            # Generic case: B x T x C -> B x C x T
            x = x.transpose(1, 2)

            x = self.convolution_1.inference(x)

            # Feed conv output to attention layer as query
            # (B x T x C)
            x = x.transpose(1, 2)
            x, alignment = self.attention_1(x, keys, values, prev_max_attention_idx[0])
            prev_max_attention_idx[0] = alignment[0].data[0].max(-1)[1].cpu().numpy()[0]

            # (B x C x T)
            x = x.transpose(1, 2)
            x = self.convolution_2.inference(x)

            x = self.convolution_3.inference(x)

            x = self.convolution_4.inference(x)

            x = x.transpose(1, 2)
            x, alignment = self.attention_4(x, keys, values, prev_max_attention_idx[1])
            prev_max_attention_idx[1] = alignment[0].data[0].max(-1)[1].cpu().numpy()[0]



            # well, I'm not sure this is really necesasary
            decoder_state = x

            # TODO take a look at korean and paper it was made in a differetn way there
            # project to mel-spectorgram
            x = F.sigmoid(self.fc2(decoder_state))

            # Done flag
            done = F.sigmoid(self.fc3(decoder_state))


            decoder_states += [decoder_state]
            outputs += [x]
            alignments += [alignment]
            dones += [done]

            t += 1

            if t > 202:  # self.max_decoder_steps:
                break

        # Remove 1-element time axis
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        decoder_states = list(map(lambda x: x.squeeze(1), decoder_states))
        outputs = list(map(lambda x: x.squeeze(1), outputs))

        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 1)
        decoder_states = torch.stack(decoder_states).transpose(0, 1).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, decoder_states, alignments


class Converter(nn.Module):
    # TODO rewrite description
    r"""
    Provides functionality for decoding in a seq2seq framework with attention.

    Args:
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        n_mels (int): number of mel filters
        outputs_per_step (int): number of deocder outputs per step, it also corresponds to reduction factor
        opt (Namespace): collection of all model arguments

    Inputs: inputs, encoder_outputs
        - **inputs** (T, N, 80): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (T, N, F): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).

    Outputs: mel_specs, ret_dict
        - **mel_specs** (T, N, 80): decoded mel spectrograms.
        - **attention**: (T_encoder, T_decoder): attention matrix
    """

    def __init__(self, config):
        super(Converter, self).__init__()

        self.decoder_n_hidden = config.decoder_n_hidden
        self.reduction_factor = config.reduction_factor
        self.max_spec_length = config.max_spec_length
        dropout_prob = config.dropout_prob
        converter_conv_width = config.converter_conv_width
        n_fft = config.n_fft

        self.converter_conv_blocks = nn.Sequential(
            ConvBlock(self.decoder_n_hidden // self.reduction_factor, kernel_size=converter_conv_width, dropout_prob=dropout_prob),
            ConvBlock(self.decoder_n_hidden // self.reduction_factor, kernel_size=converter_conv_width, dropout_prob=dropout_prob),
            ConvBlock(self.decoder_n_hidden // self.reduction_factor, kernel_size=converter_conv_width, dropout_prob=dropout_prob),
            ConvBlock(self.decoder_n_hidden // self.reduction_factor, kernel_size=converter_conv_width, dropout_prob=dropout_prob),
            ConvBlock(self.decoder_n_hidden // self.reduction_factor, kernel_size=converter_conv_width, dropout_prob=dropout_prob)
        )
        self.converter_fc = nn.utils.weight_norm(nn.Linear(self.decoder_n_hidden // self.reduction_factor, n_fft // 2 + 1), dim=None)

        #TODO deal with cuda
        # self.cuda()

    def forward(self, decoder_out):
        converter_input = decoder_out.contiguous().view(-1, self.max_spec_length, self.decoder_n_hidden // self.reduction_factor)
        lin_spec = converter_input.transpose(1, 2)  # switch to tensor[batch_size, num_channels, num_time_stamps]
        lin_spec = self.converter_conv_blocks(lin_spec)
        lin_spec = lin_spec.transpose(1, 2)  # switch to tensor[batch_size, num_timestamps, num_channels]

        lin_spec = self.converter_fc(lin_spec)

        return lin_spec