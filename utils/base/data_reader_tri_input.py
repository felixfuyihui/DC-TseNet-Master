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
import numpy as np

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from base.misc import check, read_path, read_key
from sigproc.sigproc import wavread

class DataReader(object):
    """Data reader for evaluation."""

    def __init__(self, mix_c1_scp, s1_c1_scp, s2_c1_scp, mix_c2_scp, s1_c2_scp, s2_c2_scp):
        """Initialize DataReader. (2 mixtures)

        Args:
            mix_scp: scp file for mixed waves (KALDI format)
        """
        check(mix_c1_scp, s1_c1_scp, s2_c1_scp, mix_c2_scp, s1_c2_scp, s2_c2_scp)
        self.key = read_key(mix_c1_scp)
        self.mix_c1_path = read_path(mix_c1_scp)
        self.s1_c1_path = read_path(s1_c1_scp)
        self.s2_c1_path = read_path(s2_c1_scp)
        self.mix_c2_path = read_path(mix_c2_scp)
        self.s1_c2_path = read_path(s1_c2_scp)
        self.s2_c2_path = read_path(s2_c2_scp)

    def __len__(self):
        return len(self.mix_c1_path)

    def read(self):
        for i in range(len(self.mix_c1_path)):
            key = self.key[i]
            mix_c1_sample = wavread(self.mix_c1_path[i])[0]
            s1_c1_sample = wavread(self.s1_c1_path[i])[0]
            s2_c1_sample = wavread(self.s2_c1_path[i])[0]
            mix_c2_sample = wavread(self.mix_c2_path[i])[0]
            s1_c2_sample = wavread(self.s1_c2_path[i])[0]
            s2_c2_sample = wavread(self.s2_c2_path[i])[0]
            # mix_c1_sample = mix_c1_sample[:30000]
            # mix_c2_sample = mix_c2_sample[:30000]
            mix_sample = np.stack(
            (mix_c1_sample - mix_c2_sample, 
             # mix_c1_sample + mix_c2_sample,
             mix_c1_sample,
             mix_c2_sample,
            ), axis=0).squeeze(-1)
            sample = {
                'key': key,
                'mix': torch.from_numpy(mix_sample.reshape(1, 3, -1)),
                's1': torch.from_numpy(s1_c1_sample.reshape(1, 1, -1)),
                's2': torch.from_numpy(s2_c1_sample.reshape(1, 1, -1)),
            }
            yield sample
