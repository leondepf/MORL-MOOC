
import numpy as np
from numpy import array
import random
from collections import deque
import os


def get_padding_sequence(sequence, t):
    size = sequence.shape[0]
    seq = sequence[:t]
    seq = np.append(seq, np.zeros((size-t,sequence.shape[1])),axis=0)
    return seq

def get_padding_sequence_batched(sequence, t):
    size = sequence.shape[1]
    seq = sequence[:,:t]
    seq = np.append(seq, np.zeros((sequence.shape[0],size-t,sequence.shape[2])),axis=1)
    return seq