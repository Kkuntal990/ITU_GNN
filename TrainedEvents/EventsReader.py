#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:12:47 2020

@author: aayush
"""

from tensorflow.python.summary.summary_iterator import summary_iterator


# This example supposes that the events file contains summaries with a
# summary value tag 'loss'.  These could have been added by calling
# `add_summary()`, passing the output of a scalar summary op created with
# with: `tf.scalar_summary(['loss'], loss_tensor)`.
for e in summary_iterator("10k/events.out.tfevents.1596635775.63d5a7fc86b5"):
    for v in e.summary.value:
        if v.tag == 'r-squared':
            print(e.step ,v.tag,v.simple_value)
        if v.tag == 'loss':
            print(e.step ,v.tag , v.simple_value)
