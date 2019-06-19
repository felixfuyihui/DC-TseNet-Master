#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright  2019  Northwestern Polytechnical University (author: Yihui Fu)


from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import

import pprint

pp = pprint.PrettyPrinter()


def str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Argument needs to be a boolean, got {}'.format(s))
    return {'true': True, 'false': False}[s.lower()]
