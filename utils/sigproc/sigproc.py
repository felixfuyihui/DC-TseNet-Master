#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2019  Northwestern Polytechnical University (author: Yihui Fu)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import soundfile as sf


def wavread(filename):
    data, sample_rate = sf.read(filename, dtype='float32')
    return data.reshape(-1, 1), sample_rate


def wavwrite(data, sample_rate, filename):
    max_value_int16 = (1 << 15) - 1
    data *= max_value_int16
    sf.write(filename, data.astype(np.int16), sample_rate, subtype='PCM_16',
             format='WAV')
