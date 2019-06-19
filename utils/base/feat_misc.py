#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2019  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def splice(utt, left, right):
    r"""Splice the utterance.
    Args:
        utt: Numpy matrix containing the utterance features to be spliced.
             shape [frames, dim]
        left: left contexts.
        right: right contexts.
    Return:
        A numpy array containing the spliced features.
    """
    # create spliced utterance holder
    utt_spliced = np.zeros([utt.shape[0], utt.shape[1] * (1 + left + right)],
                           dtype=np.float32)
    # middle part is just the uttarnce
    utt_spliced[:, left * utt.shape[1] : (left + 1) * utt.shape[1]] = utt

    for i in range(left):
        utt_spliced[
            i + 1 : utt_spliced.shape[0],
            (left - i - 1) * utt.shape[1] : (left - i) * utt.shape[1]
        ] = utt[0 : utt.shape[0] - i - 1, :]
        utt_spliced[
            0 : i + 1,
            (left - i - 1) * utt.shape[1] : (left - i) * utt.shape[1]
        ] = np.tile(utt[0, :], (i + 1, 1))
    for i in range(right):
        utt_spliced[
            0: utt_spliced.shape[0] - i - 1,
            (right + i + 1) * utt.shape[1] : (right + i + 2) * utt.shape[1]
        ] = utt[i + 1 : utt.shape[0], :]
        utt_spliced[
            utt_spliced.shape[0] - i - 1 : utt_spliced.shape[0],
            (right + i + 1) * utt.shape[1] : (right + i + 2) * utt.shape[1]
        ] = np.tile(utt[utt.shape[0] - 1, :], (i + 1, 1))
    return np.array(utt_spliced)


def apply_cmvn(utt, mean, variance, reverse=False):
    r"""Apply mean and variance normalization based on previously computed statistics.
    Args:
        utt: The utterance feature numpy matrix. [frames, dim]
    Return:
        A numpy array containing the mean and variance normalized features.
    """
    if not reverse:
        return np.divide(np.subtract(utt, mean), np.sqrt(variance))
    else:
        return np.add(np.multiply(utt, np.sqrt(variance)), mean)
