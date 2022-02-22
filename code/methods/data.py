from pathlib import Path
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from methods.utils.data_utilities import _segment_index
from utils.common import int16_samples_to_float32


class BaseDataset(Dataset):
    """ User defined datset

    """
    def __init__(self, args, cfg, dataset):
        """
        Args:
            args: input args
            cfg: configurations
            dataset: dataset used
        """
        super().__init__()

        self.args = args
        self.sample_rate = cfg['data']['logmelIV']['sample_rate']
        self.clip_length = dataset.clip_length

        # Chunklen and hoplen and segmentation. Since all of the clips are 30s long, it only segments once here
        # It's different from traing data
        data = np.zeros((1, self.clip_length * self.sample_rate))
        chunklen = int(10 * self.sample_rate)
        self.segmented_indexes, self.segmented_pad_width = _segment_index(data, chunklen, hoplen=chunklen)
        self.num_segments = len(self.segmented_indexes)

        # Data path
        data_sr_folder_name = '{}fs'.format(self.sample_rate)
        main_data_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('data').joinpath(data_sr_folder_name)
        self.data_dir = main_data_dir
        self.fn_list = list()
        for split in os.listdir(self.data_dir):
            if 'test' in str(split):
                continue
            split_list = [path.stem for path in sorted(self.data_dir.joinpath(split).glob('*.h5')) \
                if not path.name.startswith('.')]
            #sys.exit(type(split_list))
            self.fn_list.extend(split_list)           
        self.fn_list = [fn + '%' + str(n) for fn in self.fn_list for n in range(self.num_segments)]

    def __len__(self):
        """Get length of the dataset

        """
        return len(self.fn_list)

    def __getitem__(self, idx):
        """
        Read features from the dataset
        """
        fn_segment = self.fn_list[idx]
        fn, n_segment = fn_segment.split('%')[0], int(fn_segment.split('%')[1])
        data_path = self.data_dir.joinpath('dev') if fn[5]=='4' else self.data_dir.joinpath('train')
        data_path = data_path.joinpath(fn + '.h5')
        index_begin = self.segmented_indexes[n_segment][0]
        index_end = self.segmented_indexes[n_segment][1]
        pad_width_before = self.segmented_pad_width[n_segment][0]
        pad_width_after = self.segmented_pad_width[n_segment][1]
        with h5py.File(data_path, 'r') as hf:
            x = int16_samples_to_float32(hf['waveform'][:, index_begin: index_end])
        pad_width = ((0, 0), (pad_width_before, pad_width_after))                    
        x = np.pad(x, pad_width, mode='constant')
        sample = {
            'waveform': x
        }
          
        return sample    


class PinMemCustomBatch:
    def __init__(self, batch_dict):
        batch_x = []
        for n in range(len(batch_dict)):
            batch_x.append(batch_dict[n]['waveform'])
        batch_x = np.array(batch_x)
        self.batch_out_dict = {
            'waveform': torch.tensor(batch_x, dtype=torch.float32),
        }

    def pin_memory(self):
        self.batch_out_dict['waveform'] = self.batch_out_dict['waveform'].pin_memory()
        return self.batch_out_dict


def collate_fn(batch_dict):
    """
    Merges a list of samples to form a mini-batch
    Pin memory for customized dataset
    """
    return PinMemCustomBatch(batch_dict)
