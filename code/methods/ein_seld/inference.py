from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from methods.ein_seld import data
from methods.inference import BaseInferer
from tqdm import tqdm
from utils.common import track_to_list
import os

class Inferer(BaseInferer):

    def __init__(self, cfg, dataset, af_extractor, model, cuda, test_set=None):
        super().__init__()
        self.cfg = cfg
        self.af_extractor = af_extractor
        self.model = model
        self.cuda = cuda
        self.max_loc_value = dataset.max_loc_value
        self.dataset = dataset
        self.spec_channel = [0,1,2,3,7,8,9,10]

        # Scalar
        if cfg['data']['audio_feature'] == 'logmel&intensity':
            cfg_logmelIV = cfg['data']['logmelIV']
            scalar_h5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('scalar')
            fn_scalar = '{}_sr{}_nfft{}_hop{}_mel{}.h5'.format(cfg['data']['audio_feature'], 
                cfg_logmelIV['sample_rate'], cfg_logmelIV['n_fft'], cfg_logmelIV['hop_length'], cfg_logmelIV['n_mels'])
            scalar_path = scalar_h5_dir.joinpath(fn_scalar)
            with h5py.File(scalar_path, 'r') as hf:
                if self.cfg['data']['num_foa'] == 1:
                    self.mean = hf['mean'][:][:,:7]
                    self.std = hf['std'][:][:,:7]
                elif self.cfg['data']['num_foa'] == 2:
                    self.mean = hf['mean'][:]
                    self.std = hf['std'][:]
        elif cfg['data']['audio_feature'] == 'SALSA':
            feature_description = test_set.feature_description + 'feature_scaler.h5'
            feature_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('features')
            scalar_path = Path(feature_dir).joinpath(test_set.feature_type).joinpath(feature_description)
            with h5py.File(scalar_path, 'r') as hf:
                if self.cfg['data']['num_foa'] == 1:
                    self.mean = hf['mean'][:][:4][None,...]
                    self.std = hf['std'][:][:4][None,...]
                elif self.cfg['data']['num_foa'] == 2:
                    self.mean = hf['mean'][:][None,...]
                    self.std = hf['std'][:][None,...]
        if cuda:
            self.mean = torch.tensor(self.mean, dtype=torch.float32).cuda()
            self.std = torch.tensor(self.std, dtype=torch.float32).cuda()
        
        self.clip_length = dataset.clip_length
        self.label_resolution = dataset.label_resolution
        
    def infer(self, generator):
        fn_list, n_segment_list = [], []
        pred_sed_list, pred_doa_list = [], []

        iterator = tqdm(generator)
        for batch_sample in iterator:
            batch_x = batch_sample['data']
            if self.cuda:
                batch_x = batch_x.cuda(non_blocking=True)
            with torch.no_grad():
                if self.af_extractor:
                    self.af_extractor.eval()
                self.model.eval()
                
                if self.cfg['data']['audio_feature'] == 'logmel&intensity':
                    if self.cfg['data']['num_foa'] == 2:
                        batch_x_A = self.af_extractor(batch_x[:,:4])
                        batch_x_B = self.af_extractor(batch_x[:,4:])
                        batch_x = torch.cat((batch_x_A, batch_x_B),axis=1)
                        batch_x = (batch_x - self.mean) / self.std
                    elif self.cfg['data']['num_foa'] == 1:
                        batch_x = self.af_extractor(batch_x[:,:4])
                        batch_x = (batch_x - self.mean) / self.std
                elif self.cfg['data']['audio_feature'] == 'SALSA':
                    if self.cfg['data']['num_foa'] == 2:
                        batch_x[:, self.spec_channel] = (batch_x[:, self.spec_channel] - self.mean) / self.std
                    elif self.cfg['data']['num_foa'] == 1:
                        batch_x[:,:4] = (batch_x[:,:4] - self.mean) / self.std

                pred = self.model(batch_x)
                pred['sed'] = torch.sigmoid(pred['sed'])
            fn_list.append(batch_sample['filename'])
            n_segment_list.append(batch_sample['n_segment'])
            pred_sed_list.append(pred['sed'].cpu().detach().numpy())
            pred_doa_list.append(pred['doa'].cpu().detach().numpy())

        iterator.close()

        self.fn_list = [fn for row in fn_list for fn in row]
        self.n_segment_list = [n_segment for row in n_segment_list for n_segment in row]
        pred_sed = np.concatenate(pred_sed_list, axis=0)
        pred_doa = np.concatenate(pred_doa_list, axis=0)

        self.num_segments = max(self.n_segment_list) + 1
        origin_num_clips = int(pred_sed.shape[0]/self.num_segments)
        origin_T = int(pred_sed.shape[1]*self.num_segments)
        pred_sed = pred_sed.reshape((origin_num_clips, origin_T, 3, -1))[:, :int(self.clip_length / self.label_resolution)]
        pred_doa = pred_doa.reshape((origin_num_clips, origin_T, 3, -1))[:, :int(self.clip_length / self.label_resolution)]

        pred = {
            'sed': pred_sed,
            'doa': pred_doa
        }
        return pred

    def fusion(self, submissions_dir, predictions_dir, preds):
        """ Average ensamble predictions

        """
        num_preds = len(preds)
        pred_sed = []
        pred_doa = []
        for n in range(num_preds):
            pred_sed.append(preds[n]['sed'])
            pred_doa.append(preds[n]['doa'])
        pred_sed = np.array(pred_sed).mean(axis=0) # Ensemble
        pred_doa = np.array(pred_doa).mean(axis=0) # Ensemble
        fn_list = self.fn_list[::self.num_segments]

        prediction_path = predictions_dir.joinpath('predictions.h5')
        with h5py.File(prediction_path, 'w') as hf:
            hf.create_dataset(name='sed', data=pred_sed, dtype=np.float32)
            hf.create_dataset(name='doa', data=pred_doa, dtype=np.float32)
            # print(fn_list)
        N, T = pred_sed.shape[:2]
        pred_sed_max = pred_sed.max(axis=-1)
        pred_sed_max_idx = pred_sed.argmax(axis=-1)
        pred_sed = np.zeros_like(pred_sed)
        for b_idx in range(N):
            for t_idx in range(T):
                for track_idx in range(3):
                    pred_sed[b_idx, t_idx, track_idx, pred_sed_max_idx[b_idx, t_idx, track_idx]] = \
                        pred_sed_max[b_idx, t_idx, track_idx]
        pred_sed = (pred_sed > self.cfg['inference']['threshold_sed']).astype(np.float32)
        
        
        for clip in range(N):
            fn = fn_list[clip]
            pred_sed_f = pred_sed[clip][None, ...]
            pred_doa_f = pred_doa[clip][None, ...]
            pred_list = track_to_list(pred_sed_f, pred_doa_f, self.max_loc_value)
            csv_path = submissions_dir.joinpath(fn + '.csv')
            self.write_submission(self.dataset.label_dic_task2, csv_path, pred_list)
        print('Rsults are saved to {}\n'.format(str(submissions_dir)))


