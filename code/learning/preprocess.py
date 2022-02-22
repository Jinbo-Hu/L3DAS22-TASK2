import os
import shutil
from pathlib import Path
from timeit import default_timer as timer

import h5py
import librosa
import numpy as np
import pandas as pd
import torch
from methods.data import BaseDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.common import float_samples_to_int16, find_key_from_value
from utils.config import get_afextractor


class Preprocessor_task2:
    
    """Preprocess the audio data.

    1. Extract wav file and store to hdf5 file
    2. Extract meta file and store to hdf5 file
    """
    
    def __init__(self, args, cfg, dataset):
        """
        Args:
            args: parsed args
            cfg: configurations
            dataset: dataset class
        """
        self.args = args
        self.cfg = cfg
        self.dataset = dataset
        self.cfg_logmelIV = cfg['data']['logmelIV']

        # Path for dataset
        self.hdf5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2')

        # Path for extraction of wav
        self.data_dir_list = [
            dataset.dataset_dir['task2']['dev'].joinpath('data'),
            dataset.dataset_dir['task2']['train'].joinpath('data'),
            dataset.dataset_dir['task2']['test'].joinpath('data')
        ]
        data_h5_dir = self.hdf5_dir.joinpath('data').joinpath('{}fs'.format(self.cfg_logmelIV['sample_rate']))
        self.data_h5_dir_list = [
            data_h5_dir.joinpath('dev'),
            data_h5_dir.joinpath('train'),
            data_h5_dir.joinpath('test')
        ]

        # Path for extraction of scalar
        
        self.scalar_h5_dir = self.hdf5_dir.joinpath('scalar')
        fn_scalar = '{}_sr{}_nfft{}_hop{}_mel{}.h5'.format(cfg['data']['audio_feature'], 
            self.cfg_logmelIV['sample_rate'], self.cfg_logmelIV['n_fft'], self.cfg_logmelIV['hop_length'], self.cfg_logmelIV['n_mels'])
        self.scalar_path = self.scalar_h5_dir.joinpath(fn_scalar)

        # Path for extraction of meta
        self.label_dir_list = [
            dataset.dataset_dir['task2']['dev'].joinpath('labels'),
            dataset.dataset_dir['task2']['train'].joinpath('labels'),
            dataset.dataset_dir['task2']['test'].joinpath('labels'),
        ]

        # Path for extraction of frame label
        self.meta_frame_csv_dir_list = [
            self.hdf5_dir.joinpath('meta').joinpath('frame').joinpath('dev'),
            self.hdf5_dir.joinpath('meta').joinpath('frame').joinpath('train'),
            self.hdf5_dir.joinpath('meta').joinpath('frame').joinpath('test')
        ]

        # Path for extraction of track label
        self.meta_track_h5_dir_list = [
            self.hdf5_dir.joinpath('meta').joinpath('track').joinpath('dev'),
            self.hdf5_dir.joinpath('meta').joinpath('track').joinpath('train'),
        ]

        if args.dataset_type == 'train':
            self.data_dir_list = self.data_dir_list[:2]
            self.data_h5_dir_list = self.data_h5_dir_list[:2]
            self.label_dir_list = self.label_dir_list[:2]
            self.meta_frame_csv_dir_list = self.meta_frame_csv_dir_list[:2]
        elif args.dataset_type == 'test':
            self.data_dir_list = self.data_dir_list[2:]
            self.data_h5_dir_list = self.data_h5_dir_list[2:]
            self.label_dir_list = self.label_dir_list[2:]
            self.meta_frame_csv_dir_list = self.meta_frame_csv_dir_list[2:]


    def extract_data(self):
        """ Extract wave and store to hdf5 file

        """
        print('Converting wav file to hdf5 file starts......\n')

        for h5_dir in self.data_h5_dir_list:
            if h5_dir.is_dir():
                flag = input("HDF5 folder {} is already existed, delete it? (y/n)".format(h5_dir)).lower()
                if flag == 'y':
                    shutil.rmtree(h5_dir)
                elif flag == 'n':
                    print("User select not to remove the HDF5 folder {}. The process will quit.\n".format(h5_dir))
                    return
            h5_dir.mkdir(parents=True)
        

        for idx, data_dir in enumerate(self.data_dir_list):
            h5_dir = self.data_h5_dir_list[idx]
            data_path = os.listdir(data_dir)
            data_path_A = [i for i in data_path if i.split('.')[0].split('_')[-1]=='A']
            audio_count = 0
            for wav_file_A in data_path_A:
                wav_file_B = wav_file_A[:-5] + 'B' +  wav_file_A[-4:]  #change A with B
                wav_path_A = data_dir.joinpath(wav_file_A)
                wav_path_B = data_dir.joinpath(wav_file_B)
                data_A, _ = librosa.load(wav_path_A, sr=self.cfg_logmelIV['sample_rate'], mono=False)
                data_B, _ = librosa.load(wav_path_B, sr=self.cfg_logmelIV['sample_rate'], mono=False)

                # stack two ambisonics data
                data = np.concatenate((data_A, data_B), axis=0)

                # save to h5py
                h5_file = wav_file_A.replace('_A','').replace('.wav','.h5')
                h5_path = h5_dir.joinpath(h5_file)
                with h5py.File(h5_path, 'w') as hf:
                        hf.create_dataset(name='waveform', data=float_samples_to_int16(data), dtype=np.int16)

                audio_count += 1

                print('{}, {}, {}'.format(audio_count, h5_path, data.shape))


    def extract_frame_label(self):
        """ Extract frame label for evaluating. Store to csv file.

        """
        num_frames = int(self.dataset.clip_length / self.dataset.label_resolution)

        print('Converting meta file to frame label file starts......\n')

        for meta_frame_dir in self.meta_frame_csv_dir_list:
            if meta_frame_dir.is_dir():
                flag = input("frame label folder {} is already existed, delete it? (y/n)".format(meta_frame_dir)).lower()
                if flag == 'y':
                    shutil.rmtree(meta_frame_dir)
                elif flag == 'n':
                    print("User select not to remove the frame label folder {}. The process will quit.\n".format(meta_frame_dir))
                    return
        #quantize time stamp to step resolution
        quantize = lambda x: round(float(x) / self.dataset.label_resolution)
        
        for idx, label_dir in enumerate(self.label_dir_list): # label dir
            label_list = os.listdir(label_dir)
            self.meta_frame_csv_dir_list[idx].mkdir(parents=True, exist_ok=True)
            iterator = tqdm(enumerate(label_list), total=len(label_list), unit='it')
            for idy, path in iterator: # label path
                frame_label = {}
                for i in range(num_frames):
                    frame_label[i] = []
                path = label_dir.joinpath(path)
                df = pd.read_csv(path)
                meta_path = self.meta_frame_csv_dir_list[idx].joinpath(path.stem + '.csv')
                for idz, row in df.iterrows():
                    #compute start and end frame position (quantizing)
                    start = quantize(row['Start'])
                    end = quantize(row['End'])
                    start_frame = int(start)
                    end_frame = int(end)
                    class_id = self.dataset.label_dic_task2[row['Class']]  #int ID of sound class name
                    sound_frames = np.arange(start_frame, end_frame)
                    for f in sound_frames:
                        local_frame_label = [class_id, row['X'], row['Y'],row['Z'], idz]
                        frame_label[f].append(local_frame_label)
                for frame in range(num_frames):
                    if frame_label[frame]:
                        for event in frame_label[frame]:
                            event[0] = find_key_from_value(self.dataset.label_dic_task2, event[0])[0]
                            with meta_path.open('a') as f:
                                f.write('{},{},{},{},{},{}\n'.format(frame, event[0], event[1], event[2], event[3], event[4]))   
                tqdm.write('{}, {}'.format(idy, meta_path))


    def extract_track_label(self):
        """ Extract track label for permutation invariant training. Store to h5 file

        """
        num_tracks = self.dataset.max_ov
        num_frames = int(self.dataset.clip_length / self.dataset.label_resolution)
        num_classes = self.dataset.num_classes

        #quantize time stamp to step resolution
        quantize = lambda x: round(float(x) / self.dataset.label_resolution)

        for idx, label_dir in enumerate(self.label_dir_list):
            label_list = os.listdir(label_dir)
            self.meta_track_h5_dir_list[idx].mkdir(parents=True, exist_ok=True)
            iterator = tqdm(enumerate(label_list), total=len(label_list), unit='it')
            for idy, path in iterator:
                sed_label = np.zeros((num_frames, num_tracks, num_classes))
                doa_label = np.zeros((num_frames, num_tracks, 3))
                path = label_dir.joinpath(path)
                df = pd.read_csv(path)
                for idz, row in df.iterrows():
                    #compute start and end frame position (quantizing)
                    start = quantize(row['Start'])
                    end = quantize(row['End'])
                    start_frame = int(start)
                    end_frame = int(end)
                    class_id = self.dataset.label_dic_task2[row['Class']]  #int ID of sound class name
                    for track_idx in range(num_tracks):
                        if sed_label[start_frame][track_idx].sum() == 0:
                            sed_label[start_frame:end_frame, track_idx, class_id] = 1
                            doa_label[start_frame:end_frame, track_idx, 0] = row['X'] 
                            doa_label[start_frame:end_frame, track_idx, 1] = row['Y']
                            doa_label[start_frame:end_frame, track_idx, 2] = row['Z'] 
                            break
                        else:
                            track_idx += 1
                
                meta_path = self.meta_track_h5_dir_list[idx].joinpath(path.stem + '.h5')
                with h5py.File(meta_path, 'w') as hf:
                    hf.create_dataset(name='sed_label', data=sed_label, dtype=np.float32)
                    hf.create_dataset(name='doa_label', data=doa_label, dtype=np.float32)
                
                tqdm.write('{}, {}'.format(idy, meta_path))
    

    def extract_scalar(self):
        """ Extract scalar and store to hdf5 file

        """
        print('Extracting scalar......\n')
        self.scalar_h5_dir.mkdir(parents=True, exist_ok=True)
        cuda_enabled = not self.args.no_cuda and torch.cuda.is_available()
        train_set = BaseDataset(self.args, self.cfg, self.dataset)
        data_generator = DataLoader(
            dataset=train_set,
            batch_size=16,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        af_extractor = get_afextractor(self.cfg, cuda_enabled).eval()
        iterator = tqdm(enumerate(data_generator), total=len(data_generator), unit='it')
        features_A = []
        features_B = []
        begin_time = timer()
        for it, batch_sample in iterator:
            if it == len(data_generator):
                break
            batch_x_A = batch_sample['waveform'][:,:4]
            batch_x_B = batch_sample['waveform'][:,4:]
            batch_x_A.require_grad = False
            batch_x_B.require_grad = False
            if cuda_enabled:
                batch_x_A = batch_x_A.cuda(non_blocking=True)
                batch_x_B = batch_x_B.cuda(non_blocking=True)
            batch_y_A = af_extractor(batch_x_A).transpose(0, 1) # (C,N,T,F)
            batch_y_B = af_extractor(batch_x_B).transpose(0, 1) # (C,N,T,F)
            C, _, _, F = batch_y_A.shape
            features_A.append(batch_y_A.reshape(C, -1, F).cpu().numpy()) # (C, N*T, F)
            features_B.append(batch_y_B.reshape(C, -1, F).cpu().numpy()) # (C, N*T, F)
    
        iterator.close()
        features_A = np.concatenate(features_A, axis=1)
        features_B = np.concatenate(features_B, axis=1)
        mean_A = []
        mean_B = []
        std_A = []
        std_B = []
        for ch in range(C):
            mean_A.append(np.mean(features_A[ch], axis=0, keepdims=True))
            std_A.append(np.std(features_A[ch], axis=0, keepdims=True))
            mean_B.append(np.mean(features_B[ch], axis=0, keepdims=True))
            std_B.append(np.std(features_B[ch], axis=0, keepdims=True))
        mean_A = np.stack(mean_A)[None, ...]
        std_A = np.stack(std_A)[None, ...]
        mean_B = np.stack(mean_B)[None, ...]
        std_B = np.stack(std_B)[None, ...]
        mean = np.concatenate((mean_A, mean_B), axis=1)
        std = np.concatenate((std_A, std_B), axis=1)

        # save to h5py
        with h5py.File(self.scalar_path, 'w') as hf:
            hf.create_dataset(name='mean', data=mean, dtype=np.float32)
            hf.create_dataset(name='std', data=std, dtype=np.float32)
        print("\nScalar saved to {}\n".format(str(self.scalar_path)))
        print("Extacting scalar finished! Time spent: {:.3f} s\n".format(timer() - begin_time))            


