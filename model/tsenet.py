#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2019  Northwestern Polytechnical University (author: Yihui Fu)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import torch
import torch.nn as nn

from model.modules import Conv1d, Conv1dBlock, ConvTranspose1d, normalization

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from model.show import show_model, show_params
from evaluate.si_sdr_torch import permute_si_sdr


class TseNet(nn.Module):
    """Time-domain audio separation network."""

    def __init__(self,
                 autoencoder1_channels,
                 autoencoder1_kernel_size,
                 autoencoder2_channels,
                 autoencoder2_kernel_size,
                 autoencoder3_channels,
                 autoencoder3_kernel_size,
                 bottleneck_channels,
                 convolution_channels,
                 convolution_kernel_size,
                 num_blocks,
                 num_repeat,
                 num_speakers=2,
                 normalization_type='gLN',
                 active_func='relu'):
        super(TseNet, self).__init__()
        self.autoencoder1_channels = autoencoder1_channels
        self.autoencoder1_kernel_size = autoencoder1_kernel_size
        self.autoencoder2_channels = autoencoder2_channels
        self.autoencoder2_kernel_size = autoencoder2_kernel_size
        self.autoencoder3_channels = autoencoder3_channels
        self.autoencoder3_kernel_size = autoencoder3_kernel_size
        self.bottleneck_channels = bottleneck_channels
        self.convolution_channels = convolution_channels
        self.convolution_kernel_size = convolution_kernel_size
        self.num_blocks = num_blocks
        self.num_repeat = num_repeat
        self.autoencoder1_stride = int(self.autoencoder1_kernel_size / 2)
        self.stride = 1
        self.autoencoder2_stride = 1
        self.autoencoder3_stride = 1
        self.num_speakers = num_speakers
        self.normalization_type = normalization_type
        self.active_func = active_func

        self.encode1 = nn.Sequential(
            nn.Conv1d(3, autoencoder1_channels,
                      kernel_size=self.autoencoder1_kernel_size,
                      stride=self.autoencoder1_stride),
            nn.ReLU(),
        )

        self.encode2 = nn.Sequential(
            Conv1d(autoencoder1_channels, autoencoder2_channels,
                      kernel_size=self.autoencoder2_kernel_size,
                      stride=self.autoencoder2_stride),
            nn.ReLU(),
        )

        self.encode3 = nn.Sequential(
            Conv1d(autoencoder2_channels, autoencoder3_channels,
                      kernel_size=self.autoencoder3_kernel_size,
                      stride=self.autoencoder3_stride),
            nn.ReLU(),
        )

        self.encode_norm = normalization('cLN', autoencoder3_channels)

        self.conv1 = Conv1d(autoencoder3_channels, bottleneck_channels,
                            kernel_size=1)

        self.separation = nn.ModuleList()
        for i in range(num_repeat):
            for j in range(num_blocks):
                dilation = int(2 ** j)
                conv = Conv1dBlock(bottleneck_channels, convolution_channels,
                                   convolution_kernel_size, dilation, self.stride,
                                   self.normalization_type)
                self.separation.append(conv)

        self.conv2 = Conv1d(bottleneck_channels,
                            autoencoder3_channels * self.num_speakers,
                            kernel_size=1)
        # self.conv2 = nn.ModuleList([
        #     Conv1d(bottleneck_channels, autoencoder_channels, kernel_size=1)
        #     for _ in range(self.num_speakers)
        # ])

        # input shape [nspk, batch_size, channels, length]
        self.mask = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'softmax': nn.Softmax(dim=0),
        }[self.active_func]

        self.decode = ConvTranspose1d(
            in_channels=autoencoder1_channels,
            out_channels=1,
            kernel_size=self.autoencoder1_kernel_size,
            stride=self.autoencoder1_stride)
        show_model(self)
        show_params(self)

    def get_params(self, weight_decay):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

    def forward(self, sample, length):
        """Forward function.

        Args:
            sample: [batch_size, channels, length]
        """
        
        # print('input:', sample.shape)
        batch_size = sample.shape[0]
        # length = 64000
        sample = sample.reshape(batch_size, 3, -1)
        # print('input:', sample.shape)
        sample = sample.float()
        encode = self.encode1(sample)
        # print('----------------------encode1----------------------')
        # print('encode:', encode.shape)
        encode = self.encode2(encode)
        # print('----------------------encode2----------------------')
        # print('encode:', encode.shape)
        encode = self.encode3(encode)
        # print('----------------------encode3----------------------')
        # print('encode:', encode.shape)
        encode = self.encode_norm(encode)
        conv1 = self.conv1(encode)
        # print('conv1:', conv1.shape)
        current_layer = conv1
        for conv1d_layer in self.separation:
            current_layer = conv1d_layer(current_layer)
            # print('current_layer:', current_layer.shape)
        # conv2_buffer = []
        # for conv in self.conv2:
        #     y = conv(current_layer)
        #     conv2_buffer.append(y)
        # conv2 = torch.stack(conv2_buffer, dim=0)
        # [batch_size, nspk * channels, length]
        conv2 = self.conv2(current_layer)
        batch_size, channels, dim = conv2.shape
        assert self.autoencoder3_channels * self.num_speakers == channels
        conv2 = torch.reshape(conv2,
            (batch_size, self.num_speakers, self.autoencoder3_channels, dim))
        # [batch_size, nspk, channels, length] -> [nspk, batch_size, channels, length]
        conv2 = torch.transpose(conv2, 0, 1)
        # print('conv2:', conv2.shape)
        # [nspk, batch_size, channels, length]
        masks = self.mask(conv2)
        # print('mask:', masks.shape)
        maskings = encode * masks
        separation = []
        for i in range(self.num_speakers):
            masking = maskings[i, :, :, :]
            # print('masking:', masking.shape)
            # [batch_size, 1, length] -> [batch_size, length]
            decode = self.decode(masking, length).squeeze(dim=1)
            # print(length)
            # print('decode_before_stack:', decode.shape)
            separation.append(decode)
        # [batch_size, nspk, length]
        decode = torch.stack(separation, dim=1)
        # print('decode:', decode.shape)
        return decode

    def loss(self, output, source, device):
        # print("losssource:",source.shape)
        # print("lossoutput:",output.shape)
        batch_size = source.shape[0]
        length = 64000
        source = source.reshape(batch_size, 1, -1)
        source = source.float()
        loss = permute_si_sdr(output, source, device)
        return loss
