#!/usr/bin/env python -u
# -*- coding: utf-8 -*-


# Copyright  2019  Northwestern Polytechnical University (author: Yihui Fu)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import wave
import librosa
import numpy as np

from torch.utils.data import Dataset

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from base.misc import check, read_path
from sigproc.sigproc import wavread
def add_whitenoise(data):
        # print('1')
        wn = np.random.normal(0,1,len(data))
        # data_noise = np.where(data != 0.0, data.astype('float64') + 0.0002 * wn, 0.0).astype(np.float32)
        data_noise = data + 0.0005 * wn
        return data_noise
        
class TimeDomainDateset(Dataset):
    """Dataset class for time-domian speech separation."""

    def __init__(self,
                 mix_c1_scp,
                 s1_c1_scp,
                 s2_c1_scp,
                 mix_c2_scp,
                 s1_c2_scp,
                 s2_c2_scp,
                 sample_rate,
                 sample_clip_size=4):
        """Initialize the TimeDomainDateset. (2 mixtures)

        Args:
            mix_scp: scp file for mixed waves (KALDI format)
            s1_scp: scp file for speaker 1
            s2_scp: scp file for speaker 2
            sample_clip_size: segmental length (default: 4s)
        """
        check(mix_c1_scp, s1_c1_scp, s2_c1_scp, mix_c2_scp, s1_c2_scp, s2_c2_scp)
        self.sample_rate = sample_rate
        self.sample_clip_size = sample_clip_size
        self.segment_length = self.sample_rate * self.sample_clip_size
        #self.segment_length = 640
        self.mix_c1_path = read_path(mix_c1_scp)
        self.s1_c1_path = read_path(s1_c1_scp)
        self.s2_c1_path = read_path(s2_c1_scp)
        self.mix_c2_path = read_path(mix_c2_scp)
        self.s1_c2_path = read_path(s1_c2_scp)
        self.s2_c2_path = read_path(s2_c2_scp)

        self.retrieve_index = []
        for i in range(len(self.mix_c1_path)):
            sample_size = len(wavread(self.mix_c1_path[i])[0])
            if sample_size < self.segment_length:
                # wave length is smaller than segmental length
                if sample_size * 2 < self.segment_length:
                    continue
                self.retrieve_index.append((i, -1))
            else:
                # Cut wave into clips and restore the retrieve index
                sample_index = 0
                while sample_index + self.segment_length < sample_size:
                    self.retrieve_index.append((i, sample_index))
                    sample_index += self.segment_length
                if sample_index != sample_size - 1:
                    self.retrieve_index.append(
                            (i, sample_size - self.segment_length))

    def __len__(self):
        return len(self.retrieve_index)

    def __getitem__(self, index):
        utt_id, sample_index = self.retrieve_index[index]
        mix_c1_sample = wavread(self.mix_c1_path[utt_id])[0]
        s1_c1_sample = wavread(self.s1_c1_path[utt_id])[0]
        # s2_c1_sample = wavread(self.s2_c1_path[utt_id])[0]
        mix_c2_sample = wavread(self.mix_c2_path[utt_id])[0]
        # s1_c2_sample = wavread(self.s1_c2_path[utt_id])[0]
        # s2_c2_sample = wavread(self.s2_c2_path[utt_id])[0]
        if sample_index == -1:
            length = len(mix_c1_sample)
            stack_length = self.segment_length - length
            mix_c1_stack_sample = mix_c1_sample[: stack_length].reshape(-1, 1)
            s1_c1_stack_sample = s1_c1_sample[: stack_length].reshape(-1, 1)
            # s2_c1_stack_sample = s2_c1_sample[: stack_length].reshape(-1, 1)
            mix_c2_stack_sample = mix_c2_sample[: stack_length].reshape(-1, 1)
            # s1_c2_stack_sample = s1_c2_sample[: stack_length].reshape(-1, 1)
            # s2_c2_stack_sample = s2_c2_sample[: stack_length].reshape(-1, 1)
            mix_c1_clipped_sample = np.concatenate(
                    (mix_c1_sample, mix_c1_stack_sample), axis=0)
            s1_c1_clipped_sample = np.concatenate(
                    (s1_c1_sample, s1_c1_stack_sample), axis=0)
            # s2_c1_clipped_sample = np.concatenate(
            #         (s2_c1_sample, s2_c1_stack_sample), axis=0)
            mix_c2_clipped_sample = np.concatenate(
                    (mix_c2_sample, mix_c2_stack_sample), axis=0)
            # s1_c2_clipped_sample = np.concatenate(
            #         (s1_c2_sample, s1_c2_stack_sample), axis=0)
            # s2_c2_clipped_sample = np.concatenate(
            #         (s2_c2_sample, s2_c2_stack_sample), axis=0)
        else:
            end_index = sample_index + self.segment_length
            mix_c1_clipped_sample = mix_c1_sample[sample_index : end_index]
            s1_c1_clipped_sample = s1_c1_sample[sample_index : end_index]
            # s2_c1_clipped_sample = s2_c1_sample[sample_index : end_index]
            mix_c2_clipped_sample = mix_c2_sample[sample_index : end_index]
            # s1_c2_clipped_sample = s1_c2_sample[sample_index : end_index]
            # s2_c2_clipped_sample = s2_c2_sample[sample_index : end_index]
        whitenoise = np.random.normal(0,1,len(mix_c1_clipped_sample))
        whitenoise = whitenoise.reshape(-1,1)
        whitenoise = whitenoise.astype(np.float32)
        mix_c1_clipped_sample = mix_c1_clipped_sample + 5e-4 * whitenoise
        mix_c2_clipped_sample = mix_c2_clipped_sample + 5e-4 * whitenoise

        mix_clipped_sample = np.stack(
            (mix_c1_clipped_sample - mix_c2_clipped_sample, 
             # mix_c1_clipped_sample + mix_c2_clipped_sample,
             mix_c1_clipped_sample,
             mix_c2_clipped_sample,
            ), axis=0).squeeze(-1)

        src_clipped_sample = np.stack(
            (s1_c1_clipped_sample
             # s2_c1_clipped_sample
            ), axis=0).squeeze(-1)
        sample = {
            'mix': mix_clipped_sample.reshape(3, -1),
            'src': src_clipped_sample.reshape(1, -1),
        }
        return sample
