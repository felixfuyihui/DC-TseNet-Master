#!/usr/bin/env python -u
# -*- coding: utf-8 -*-


# Copyright  2019  Northwestern Polytechnical University (author: Yihui Fu)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch


def read_scp(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines


def check_consistency(value1, value2):
    if value1 != value2:
        raise ValueError('Mixmatch between {} and {}'.format(value1, value2))


def check(mix_c1_scp, s1_c1_scp, s2_c1_scp, mix_c2_scp, s1_c2_scp, s2_c2_scp):
    mix_c1_key = read_key(mix_c1_scp)
    s1_c1_key = read_key(s1_c1_scp)
    s2_c1_key = read_key(s2_c1_scp)
    check_consistency(mix_c1_key, s1_c1_key)
    check_consistency(mix_c1_key, s2_c1_key)
    mix_c2_key = read_key(mix_c2_scp)
    s1_c2_key = read_key(s1_c2_scp)
    s2_c2_key = read_key(s2_c2_scp)
    check_consistency(mix_c2_key, s1_c2_key)
    check_consistency(mix_c2_key, s2_c2_key)


def read_key(scp_file):
    scp_lst = read_scp(scp_file)
    buffer_key = []
    for line in scp_lst:
        key, path = line.strip().split()
        buffer_key.append(key)
    return buffer_key


def read_path(scp_file):
    scp_lst = read_scp(scp_file)
    buffer_path = []
    for line in scp_lst:
        key, path = line.strip().split()
        buffer_path.append(path)
    return buffer_path
