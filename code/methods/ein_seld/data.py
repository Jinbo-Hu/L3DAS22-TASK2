from pathlib import Path
from timeit import default_timer as timer
import sys

import h5py
import numpy as np
import torch
from methods.utils.data_utilities import _segment_index
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from utils.common import int16_samples_to_float32, csv_to_list
import torch.distributed as dist

class UserDataset(Dataset):
    """ User defined datset

    """
    def __init__(self, cfg, dataset, dataset_type='train'):
        """
        Args:
            args: input args
            cfg: configurations
            dataset: dataset used
            dataset_type: 'train' | 'dev' | 'dev_test' | 'test_test' | 'train_test'. 
                'train' and 'dev' are only used while training. 
                'dev_test' , 'test_test', and 'train_test' are only used while infering.
        """
        super().__init__()

        self.cfg = cfg
        self.dataset_type = dataset_type
        self.clip_length = dataset.clip_length 
        self.label_resolution = dataset.label_resolution
        self.frame_length = int(self.clip_length / self.label_resolution)
        self.split_str_index = dataset.split_str_index
        self.max_ov = dataset.max_ov
        self.num_classes = dataset.num_classes
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.max_loc_value = dataset.max_loc_value
        self.audio_feature = cfg['data']['audio_feature']

        # Chunklen and hoplen and segmentation. Since all of the clips are 30s long, it only segments once here
        if self.audio_feature == 'logmel&intensity':
            self.sample_rate = cfg['data']['logmelIV']['sample_rate']
            data = np.zeros((1, self.clip_length * self.sample_rate))
            if 'train' in self.dataset_type:
                chunklen = int(cfg['data']['train_chunklen_sec'] * self.sample_rate)     
                hoplen = int(cfg['data']['train_hoplen_sec'] * self.sample_rate)
                self.segmented_indexes, self.segmented_pad_width = _segment_index(data, chunklen, hoplen)
            elif self.dataset_type in ['dev', 'dev_test', 'test_test']:
                chunklen = int(cfg['data']['test_chunklen_sec'] * self.sample_rate)
                hoplen = int(cfg['data']['test_hoplen_sec'] * self.sample_rate)
                self.segmented_indexes, self.segmented_pad_width = _segment_index(data, chunklen, hoplen, last_frame_always_paddding=True)
            data_sr_folder_name = '{}fs'.format(self.sample_rate)
            main_data_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('data').joinpath(data_sr_folder_name)
        elif self.audio_feature == 'SALSA':
            cfg_SALSA = cfg['data']['SALSA']
            fs = cfg_SALSA['sample_rate']
            n_fft = cfg_SALSA['n_fft']
            hop_length = cfg_SALSA['hop_length']
            fmax = cfg_SALSA['fmax']
            cond_num = cfg_SALSA['cond_num']
            frames_per_clips = int(fs * self.clip_length / hop_length) 
            data = np.zeros((1, frames_per_clips))
            if 'train' in self.dataset_type:
                self.frames_per_prediction = int(self.label_resolution / (hop_length / fs)) 
                chunklen = int(cfg['data']['train_chunklen_sec'] / self.label_resolution * self.frames_per_prediction) 
                hoplen = int(cfg['data']['train_hoplen_sec'] / self.label_resolution * self.frames_per_prediction) 
                self.segmented_indexes, self.segmented_pad_width = _segment_index(data, chunklen, hoplen)
            elif self.dataset_type in ['dev', 'dev_test', 'test_test']:
                self.frames_per_prediction = int(self.label_resolution / (hop_length / fs)) 
                chunklen = int(cfg['data']['test_chunklen_sec'] / self.label_resolution * self.frames_per_prediction) 
                hoplen = int(cfg['data']['test_hoplen_sec'] / self.label_resolution * self.frames_per_prediction) 
                self.segmented_indexes, self.segmented_pad_width = _segment_index(data, chunklen, hoplen, last_frame_always_paddding=True)    
            self.feature_description = '{}fs_{}nfft_{}nhop_{}cond_{}fmaxdoa_nocompress'.format(
                fs, n_fft, hop_length, int(cond_num), int(fmax))
            self.feature_type = 'tfmap'
            main_data_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('features')\
                .joinpath(self.feature_type).joinpath(self.feature_description)
        self.num_segments = len(self.segmented_indexes)

        frame_meta_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('meta').joinpath('frame')
        track_meta_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('meta').joinpath('track')

        # paths_list: data path and n_th segment  
        if self.dataset_type == 'train':
            train_set = cfg['training']['train_set']
            assert train_set == 'train' or train_set == 'train&dev' ,\
                "train_set '{}' is not 'train' or 'train&dev'".format(train_set)
            data_dirs = [main_data_dir.joinpath('train')] if train_set == 'train' else \
                [main_data_dir.joinpath('train'), main_data_dir.joinpath('dev')]
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                if not path.name.startswith('.')]
            self.meta_dir = track_meta_dir
        elif self.dataset_type == 'dev':
            val_set = cfg['training']['valid_set']
            data_dirs = [main_data_dir.joinpath('dev')] if val_set == 'dev' else []
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                if not path.name.startswith('.')]
            # each gpu use different sampler (the same as DistributedSampler)
            self.paths_list = self.paths_list[self.rank::self.num_replicas] 
            self.meta_dir = track_meta_dir
            self.dev_gt_seld_labels = []
            for path in self.paths_list:
                frame_meta_path = frame_meta_dir.joinpath('dev').joinpath('label_'+path.stem+'.csv')
                label = csv_to_list(frame_meta_path, dataset)
                self.dev_gt_seld_labels.extend(label)
        elif self.dataset_type == 'dev_test':
            data_dirs = [main_data_dir.joinpath('dev')]
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                if not path.name.startswith('.')]
            self.meta_dir = track_meta_dir
        elif self.dataset_type == 'test_test':
            data_dirs = [main_data_dir.joinpath('test')]
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                if not path.name.startswith('.')]
        elif self.dataset_type == 'train_test':
            data_dirs = [main_data_dir.joinpath('train')]
            self.paths_list = [path for data_dir in data_dirs for path in sorted(data_dir.glob('*.h5')) \
                if not path.name.startswith('.')]
            self.meta_dir = track_meta_dir
        self.paths_list = [Path(str(path) + '%' + str(n)) for path in self.paths_list for n in range(self.num_segments)]


    def __len__(self):
        """Get length of the dataset

        """
        return len(self.paths_list)

    def __getitem__(self, idx):
        """
        Read features from the dataset
        """
        path = self.paths_list[idx]
        fn, n_segment = path.stem, int(path.name.split('%')[1])
        data_path = Path(str(path).split('%')[0])   
        index_begin = self.segmented_indexes[n_segment][0]
        index_end = self.segmented_indexes[n_segment][1]
        pad_width_before = self.segmented_pad_width[n_segment][0]
        pad_width_after = self.segmented_pad_width[n_segment][1]
        if self.audio_feature == 'logmel&intensity':
            with h5py.File(data_path, 'r') as hf:
                x = int16_samples_to_float32(hf['waveform'][:, index_begin: index_end]) if self.cfg['data']['num_foa'] == 2\
                    else int16_samples_to_float32(hf['waveform'][:4, index_begin: index_end])
            pad_width = ((0, 0), (pad_width_before, pad_width_after))
        elif self.audio_feature == 'SALSA':
            with h5py.File(data_path, 'r') as hf:
                x = hf['feature'][:, index_begin: index_end] if self.cfg['data']['num_foa'] == 2\
                    else hf['feature'][:7, index_begin: index_end]
            pad_width = ((0, 0), (0, 0), (pad_width_before, pad_width_after))
        x = np.pad(x, pad_width, mode='constant')

        if 'test' not in self.dataset_type:
            ov = fn[-3]
            if self.audio_feature == 'logmel&intensity':
                index_begin_label = int(index_begin / (self.sample_rate * self.label_resolution))
                index_end_label = int(index_end / (self.sample_rate * self.label_resolution))
                pad_width_after_label = int(pad_width_after / (self.sample_rate * self.label_resolution))
            elif self.audio_feature == 'SALSA':
                index_begin_label = int(index_begin / self.frames_per_prediction)
                index_end_label = int(index_end / self.frames_per_prediction)
                pad_width_after_label = int(pad_width_after / self.frames_per_prediction)
            meta_path = self.meta_dir.joinpath('dev').joinpath('label_'+ fn + '.h5') if fn[self.split_str_index] == '4'\
                else self.meta_dir.joinpath('train').joinpath('label_'+ fn + '.h5')
            with h5py.File(meta_path, 'r') as hf:
                sed_label = hf['sed_label'][index_begin_label: index_end_label, ...]
                doa_label = hf['doa_label'][index_begin_label: index_end_label, ...]
            if pad_width_after_label != 0:                
                sed_label_new = np.zeros((pad_width_after_label, self.max_ov, self.num_classes))
                sed_label = np.concatenate((sed_label, sed_label_new), axis=0)
                doa_label_new = np.zeros((pad_width_after_label, self.max_ov, 3))
                doa_label = np.concatenate((doa_label, doa_label_new), axis=0)
            doa_label[:,:,0] /= self.max_loc_value[0]
            doa_label[:,:,1] /= self.max_loc_value[1]
            doa_label[:,:,2] /= self.max_loc_value[2]
        if 'test' not in self.dataset_type:
            sample = {
                'filename': fn,
                'n_segment': n_segment,
                'ov': ov,
                'data': x,
                'sed_label': sed_label,
                'doa_label': doa_label,
            }
        else:
            sample = {
                'filename': fn,
                'n_segment': n_segment,
                'data': x
            }
          
        return sample    


class UserBatchSampler(Sampler):
    """User defined batch sampler. Only for train set.

    """
    def __init__(self, clip_num, batch_size, seed=2021, drop_last=False):
        self.clip_num = clip_num
        self.batch_size = batch_size
        self.random_state = None
        self.indexes = np.arange(self.clip_num)
        self.pointer = 0
        self.epoch = 0
        self.drop_last = drop_last
        self.seed = seed
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.random_state = np.random.RandomState(self.seed+self.epoch)
        self.random_state.shuffle(self.indexes)
        if not self.drop_last:
            if self.clip_num % (self.batch_size*self.num_replicas) != 0:
                padding_size = self.batch_size*self.num_replicas - self.clip_num % (self.batch_size*self.num_replicas)
                self.indexes = np.append(self.indexes, self.indexes[:padding_size])
                self.clip_num = self.clip_num + padding_size

    
    def get_state(self):
        sampler_state = {
            'random': self.random_state.get_state(),
            'indexes': self.indexes,
            'pointer': self.pointer
        }
        return sampler_state

    def set_state(self, sampler_state):
        self.random_state.set_state(sampler_state['random'])
        self.indexes = sampler_state['indexes']
        self.pointer = sampler_state['pointer']
    
    def __iter__(self):
        """
        Return: 
            batch_indexes (int): indexes of batch
        """   
        while True:
            if self.pointer >= self.clip_num:
                self.pointer = 0
                self.random_state.shuffle(self.indexes)
            
            batch_indexes = self.indexes[self.pointer: self.pointer + self.batch_size * self.num_replicas]
            self.pointer += self.batch_size * self.num_replicas
            batch_indexes = batch_indexes[self.rank:self.clip_num:self.num_replicas]
            yield batch_indexes

    def __len__(self):
        return (self.clip_num + self.num_replicas * self.batch_size - 1) // (self.num_replicas * self.batch_size)


class PinMemCustomBatch:
    def __init__(self, batch_dict):
        batch_fn = []
        batch_n_segment = []
        batch_ov = []
        batch_x = []
        batch_sed_label = []
        batch_doa_label = []
        
        for n in range(len(batch_dict)):
            batch_fn.append(batch_dict[n]['filename'])
            batch_n_segment.append(batch_dict[n]['n_segment'])
            batch_ov.append(batch_dict[n]['ov'])
            batch_x.append(batch_dict[n]['data'])
            batch_sed_label.append(batch_dict[n]['sed_label'])
            batch_doa_label.append(batch_dict[n]['doa_label'])
        
        batch_x = np.array(batch_x,)
        batch_sed_label = np.array(batch_sed_label)
        batch_doa_label = np.array(batch_doa_label)
        self.batch_out_dict = {
            'filename': batch_fn,
            'n_segment': batch_n_segment,
            'ov': batch_ov,
            'data': torch.tensor(batch_x, dtype=torch.float32),
            'sed_label': torch.tensor(batch_sed_label, dtype=torch.float32),
            'doa_label': torch.tensor(batch_doa_label, dtype=torch.float32),
        }

    def pin_memory(self):
        self.batch_out_dict['data'] = self.batch_out_dict['data'].pin_memory()
        self.batch_out_dict['sed_label'] = self.batch_out_dict['sed_label'].pin_memory()
        self.batch_out_dict['doa_label'] = self.batch_out_dict['doa_label'].pin_memory()
        return self.batch_out_dict


def collate_fn(batch_dict):
    """
    Merges a list of samples to form a mini-batch
    Pin memory for customized dataset
    """
    return PinMemCustomBatch(batch_dict)


class PinMemCustomBatchTest:
    def __init__(self, batch_dict):
        batch_fn = []
        batch_n_segment = []
        batch_x = []
        
        for n in range(len(batch_dict)):
            batch_fn.append(batch_dict[n]['filename'])
            batch_n_segment.append(batch_dict[n]['n_segment'])
            batch_x.append(batch_dict[n]['data'])
        batch_x = np.array(batch_x)
        self.batch_out_dict = {
            'filename': batch_fn,
            'n_segment': batch_n_segment,
            'data': torch.tensor(batch_x, dtype=torch.float32)
        }

    def pin_memory(self):
        self.batch_out_dict['data'] = self.batch_out_dict['data'].pin_memory()
        return self.batch_out_dict


def collate_fn_test(batch_dict):
    """
    Merges a list of samples to form a mini-batch
    Pin memory for customized dataset
    """
    return PinMemCustomBatchTest(batch_dict)
