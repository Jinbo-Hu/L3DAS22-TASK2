import numpy as np
import pandas as pd
import torch


def _segment_index(x, chunklen, hoplen, last_frame_always_paddding=False):
    """Segment input x with chunklen, hoplen parameters. Return

    Args:
        x: input, time domain or feature domain (channels, time)
        chunklen:
        hoplen:
        last_frame_always_paddding: to decide if always padding for the last frame
    
    Return:
        segmented_indexes: [(begin_index, end_index), (begin_index, end_index), ...]
        segmented_pad_width: [(before, after), (before, after), ...]
    """
    x_len = x.shape[1]

    segmented_indexes = []
    segmented_pad_width = []
    if x_len < chunklen:
        begin_index = 0
        end_index = x_len
        pad_width_before = 0
        pad_width_after = chunklen - x_len
        segmented_indexes.append((begin_index, end_index))
        segmented_pad_width.append((pad_width_before, pad_width_after))
        return segmented_indexes, segmented_pad_width

    n_frames = 1 + (x_len - chunklen) // hoplen
    for n in range(n_frames):
        begin_index = n * hoplen
        end_index = n * hoplen + chunklen
        segmented_indexes.append((begin_index, end_index))
        pad_width_before = 0
        pad_width_after = 0
        segmented_pad_width.append((pad_width_before, pad_width_after))
    
    if (n_frames - 1) * hoplen + chunklen == x_len:
        return segmented_indexes, segmented_pad_width

    # the last frame
    if last_frame_always_paddding:
        begin_index = n_frames * hoplen
        end_index = x_len
        pad_width_before = 0
        pad_width_after = chunklen - (x_len - n_frames * hoplen)        
    else:
        if x_len - n_frames * hoplen >= chunklen // 2:
            begin_index = n_frames * hoplen
            end_index = x_len
            pad_width_before = 0
            pad_width_after = chunklen - (x_len - n_frames * hoplen)
        else:
            begin_index = x_len - chunklen
            end_index = x_len
            pad_width_before = 0
            pad_width_after = 0
    segmented_indexes.append((begin_index, end_index))
    segmented_pad_width.append((pad_width_before, pad_width_after))

    return segmented_indexes, segmented_pad_width




