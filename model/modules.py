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
import torch.nn.functional as F

sys.path.append(os.path.dirname(sys.path[0]))
from model.layer_norm import CLayerNorm, GLayerNorm
class EncoderBlock(nn.Module):
    """1-D convolutional block."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride):
        """Depthwise convolution"""
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Sequential(Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride),
                                   nn.ReLU(),)
        self.conv2 = nn.Sequential(Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride),
                                   nn.ReLU(),)
        self.conv3 = nn.Sequential(Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride),
                                   nn.ReLU(),)
        self.conv4 = nn.Sequential(Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride),
                                   nn.ReLU(),)
        
    def forward(self, sample):
        conv = self.conv1(sample)
        conv = self.conv2(conv)
        conv = self.conv3(conv)
        conv = self.conv4(conv)
        return conv

class DecoderBlock(nn.Module):
    """1-D convolutional block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride):
        """Depthwise convolution"""
        super(DecoderBlock, self).__init__()
        self.dconv1 = ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.acti1 = nn.ReLU()
        self.dconv2 = ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.acti2 = nn.ReLU()
        self.dconv3 = ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.acti3 = nn.ReLU()
        self.dconv4 = ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.acti4 = nn.ReLU()

        
    def forward(self, sample, length):
        conv = self.dconv1(sample, length)
        conv = self.acti1(conv)
        conv = self.dconv2(conv, length)
        conv = self.acti2(conv)
        conv = self.dconv3(conv, length)
        conv = self.acti3(conv)
        conv = self.dconv4(conv, length)
        conv = self.acti4(conv)
        return conv

class Conv1dBlock(nn.Module):
    """1-D convolutional block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 stride,
                 norm_type):
        """Depthwise convolution"""
        super(Conv1dBlock, self).__init__()
        self.conv1 = nn.Sequential(
            Conv1d(in_channels, out_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = normalization(norm_type, out_channels)
        self.dconv = nn.Sequential(
            Conv1d(out_channels, out_channels, kernel_size, stride,
                   dilation=dilation, groups=out_channels),
            nn.PReLU(),
        )
        self.norm2 = normalization(norm_type, out_channels)
        self.conv2 = Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, sample):
        conv1 = self.conv1(sample)
        norm1 = self.norm1(conv1)
        dconv = self.dconv(norm1)
        norm2 = self.norm2(dconv)
        conv2 = self.conv2(norm2)
        return conv2 + sample

class Conv2dBlock(nn.Module):
    """1-D convolutional block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 stride,
                 norm_type):
        """Depthwise convolution"""
        super(Conv1dBlock, self).__init__()
        self.conv1 = nn.Sequential(
            Conv1d(in_channels, out_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = normalization(norm_type, out_channels)
        self.dconv = nn.Sequential(
            Conv1d(out_channels, out_channels, kernel_size, stride,
                   dilation=dilation, groups=out_channels),
            nn.PReLU(),
        )
        self.norm2 = normalization(norm_type, out_channels)
        self.conv2 = Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, sample):
        conv1 = self.conv1(sample)
        norm1 = self.norm1(conv1)
        dconv = self.dconv(norm1)
        norm2 = self.norm2(dconv)
        conv2 = self.conv2(norm2)
        return conv2 + sample

class Conv1dBlock_Encoder(nn.Module):
    """1-D convolutional block."""

    def __init__(self,
                 in_channels,
                 middle_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 stride):
        """Depthwise convolution"""
        super(Conv1dBlock_Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            Conv1d(in_channels, in_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.dconv = nn.Sequential(
            Conv1d(in_channels, middle_channels, kernel_size, stride,
                   dilation=dilation, groups=out_channels),
            nn.PReLU(),
        )
        self.conv2 = Conv1d(middle_channels, out_channels, kernel_size=1)

    def forward(self, sample):
        conv1 = self.conv1(sample)
        dconv = self.dconv(conv1)
        conv2 = self.conv2(dconv)
        return conv2 + sample

class Conv1dEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 middle_channels,
                 out_channels,
                 kernel_size,
                 stride):
        super(Conv1dEncoderBlock, self).__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv1d(in_channels, middle_channels, kernel_size, stride),
            nn.ReLU(),
        )
        self.conv2 =  nn.Sequential(
            nn.Conv1d(middle_channels, out_channels, kernel_size, stride),
            nn.ReLU(),
        )

    def forward(self, sample):
        conv1 = self.conv1(sample)
        conv2 = self.conv2(conv1)
        return conv2

class ConvTranspose1dBlock(nn.Module):
    """1-D Convolution Transpose Block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 norm_type):
        super(ConvTranspose1dBlock, self).__init__()
        self.conv1 = nn.Sequential(
            Conv1d(in_channels, out_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = normalization(norm_type, out_channels)
        self.dconv_trans = ConvTranspose1d(out_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride, groups=out_channels)
        self.norm2 = nn.Sequential(
                nn.PReLU(),
                normalization(norm_type, out_channels),
        )
        self.conv2 = Conv1d(out_channels, 1, kernel_size=1)

    def forward(self, sample, length):
        conv1 = self.conv1(sample)
        norm1 = self.norm1(conv1)
        dconv_trans = self.dconv_trans(norm1, length)
        norm2 = self.norm2(dconv_trans)
        conv2 = self.conv2(norm2)
        return conv2

class ConvTranspose1dReLU(nn.Module):
    def __init__(self,
                   in_channels,
                   out_channels,
                   kernel_size,
                   stride):
        super(ConvTranspose1dReLU, self).__init__()
        self.convout = ConvTranspose1d(in_channels, out_channels, 
                                       kernel_size=kernel_size, stride=stride)
        self.activate = nn.ReLU()
          
    def forward(self, sample, length):
        convout = self.convout(sample, length)
        convout = self.activate(convout)
        return convout

class ConvTranspose1dReLUBN(nn.Module):
    def __init__(self,
                   in_channels,
                   out_channels,
                   kernel_size,
                   stride):
        super(ConvTranspose1dReLUBN, self).__init__()
        self.convout = ConvTranspose1d(in_channels, out_channels, 
                                       kernel_size=kernel_size, stride=stride)
        self.activate = nn.ReLU()
        self.BN = nn.BatchNorm1d(out_channels)
          
    def forward(self, sample, length):
        convout = self.convout(sample, length)
        convout = self.activate(convout)
        convout = self.BN(convout)
        return convout

class ConvTranspose2dReLUBN(nn.Module):
    def __init__(self,
                   in_channels,
                   out_channels,
                   kernel_size,
                   stride):
        super(ConvTranspose2dReLUBN, self).__init__()
        self.convout = nn.ConvTranspose2d(in_channels, out_channels, 
                                       kernel_size=kernel_size, stride=stride)
        self.BN = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU()
          
    def forward(self, sample, length):
        convout = self.convout(sample, length)
        convout = self.BN(convout)
        convout = self.activate(convout)
        return convout

class Conv1d(nn.Conv1d):
    """Conv1d with SAME padding"""

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


def conv1d_same_padding(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1):
    """Conv1d with SAME padding"""
    length = input.size(2)
    kernel_size = weight.size(2)
    padding_dim = get_conv1d_padding_dim(length, stride, dilation, kernel_size)
    return F.conv1d(input, weight, bias, stride, padding=padding_dim,
                    dilation=dilation, groups=groups)


class ConvTranspose1d(nn.ConvTranspose1d):
    """ConvTranspose1d with SAME padding"""

    def __init__(self, *args, **kwargs):
        super(ConvTranspose1d, self).__init__(*args, **kwargs)

    def forward(self, input, length_out):
        return conv_transpose1d_same_padding(input, length_out, self.weight,
                self.bias, self.stride, self.padding, self.output_padding,
                self.groups, self.dilation)


def conv_transpose1d_same_padding(input,
                                  length_out,
                                  weight,
                                  bias=None,
                                  stride=1,
                                  padding=0,
                                  output_padding=0,
                                  groups=1,
                                  dilation=1):
    """Transpose conv with SAME padding"""
    length_in = input.size(2)
    kernel_size = weight.size(2)
    padding_dim, output_padding = get_conv_transpose1d_padding_dim(
            length_in, length_out, stride, kernel_size)
    return F.conv_transpose1d(input, weight, bias, stride, padding_dim,
                              output_padding, groups, dilation)


def normalization(norm_type, channels):
    """Normalization layers
      BN: [batch_size, channels, length]
      cLN: [batch_size, channels, length]
      gLN: [batch_size, channels, length]
    """
    norm = {
        'BN': nn.BatchNorm1d(channels),
        'cLN': CLayerNorm(channels),
        'gLN': GLayerNorm(channels),
    }[norm_type]
    return norm


def one_element_tuple_to_single_number(element):
    if isinstance(element, tuple):
        return element[0]
    else:
        return element


def get_conv1d_padding_dim(length, stride, dilation, kernel_size):
    stride = one_element_tuple_to_single_number(stride)
    dilation = one_element_tuple_to_single_number(dilation)
    padding_dim = (length - 1) * stride + 1 + dilation * (kernel_size - 1)
    padding_dim = int(math.ceil((padding_dim - length) / 2))
    return padding_dim


def get_conv_transpose1d_padding_dim(length_in,
                                     length_out,
                                     stride,
                                     kernel_size):
    stride = one_element_tuple_to_single_number(stride)
    padding_dim = (length_in - 1) * stride - length_out + kernel_size
    if padding_dim < 0:
        output_padding = 0 - padding_dim
        padding_dim = 0
    else:
        if padding_dim % 2 == 0:
            padding_dim = int(padding_dim / 2)
            output_padding = 0
        else:
            padding_dim = int(padding_dim / 2 + 1)
            output_padding = 1
    return padding_dim, output_padding
