import random
from pathlib import Path
import sys

from utils.common import track_to_list

import h5py
import numpy as np
import torch
from methods.training import BaseTrainer
import torch.distributed as dist
from utils.ddp_init import reduce_value


class Trainer(BaseTrainer):

    def __init__(self, args, cfg, dataset, af_extractor, valid_set, model, optimizer, losses, metrics):

        super().__init__()
        self.cfg = cfg
        self.af_extractor = af_extractor
        self.model = model
        self.optimizer = optimizer
        self.losses = losses
        self.metrics = metrics
        self.cuda = args.cuda
        self.spec_channel = [0,1,2,3,7,8,9,10]
        if args.cuda:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.clip_length = dataset.clip_length
        self.label_resolution = dataset.label_resolution
        self.max_loc_value = dataset.max_loc_value

        # Load ground truth for dcase metrics
        self.num_segments = valid_set.num_segments
        self.dev_gt_seld_labels = valid_set.dev_gt_seld_labels

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
            feature_description = valid_set.feature_description + 'feature_scaler.h5'
            feature_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('features')
            scalar_path = Path(feature_dir).joinpath(valid_set.feature_type).joinpath(feature_description)
            with h5py.File(scalar_path, 'r') as hf:
                if self.cfg['data']['num_foa'] == 1:
                    self.mean = hf['mean'][:][:4][None,...]
                    self.std = hf['std'][:][:4][None,...]
                elif self.cfg['data']['num_foa'] == 2:
                    self.mean = hf['mean'][:][None,...]
                    self.std = hf['std'][:][None,...]
        if args.cuda:
            self.mean = torch.tensor(self.mean, dtype=torch.float32).to(self.rank)
            self.std = torch.tensor(self.std, dtype=torch.float32).to(self.rank)


        self.init_train_losses()
    
    def init_train_losses(self):
        """ Initialize train losses

        """
        self.train_losses = {
            'loss_all': 0.,
            'loss_sed': 0.,
            'loss_doa': 0.,
        }

    def train_step(self, batch_sample, epoch_it):
        """ Perform a train step

        """
    
        batch_x = batch_sample['data']
        batch_target = {
            'sed': batch_sample['sed_label'],
            'doa': batch_sample['doa_label'],
            'ov': batch_sample['ov']
        }
        if self.cuda:
            batch_x = batch_x.to(self.rank, non_blocking=True)
            batch_target['sed'] = batch_target['sed'].to(self.rank, non_blocking=True)
            batch_target['doa'] = batch_target['doa'].to(self.rank, non_blocking=True)

        self.optimizer.zero_grad()
        if self.af_extractor:
            self.af_extractor.train()
        self.model.train()

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
        loss_dict = self.losses.calculate(pred, batch_target)
        loss_dict[self.cfg['training']['loss_type']].backward()

        self.optimizer.step()

        self.train_losses['loss_all'] += loss_dict['all']
        self.train_losses['loss_sed'] += loss_dict['sed']
        self.train_losses['loss_doa'] += loss_dict['doa']
        

    def validate_step(self, generator=None, max_batch_num=None, valid_type='train', epoch_it=0):
        """ Perform the validation on the train, valid set

        Generate a batch of segmentations each time
        """
        if valid_type == 'train':
            train_losses = self.train_losses.copy()
            self.init_train_losses()
            return train_losses

        elif valid_type == 'valid':
            pred_sed_list, pred_doa_list = [], []

            loss_all, loss_sed, loss_doa = 0., 0., 0.

            for batch_idx, batch_sample in enumerate(generator):
                if batch_idx == max_batch_num:
                    break
                batch_x = batch_sample['data']
                batch_target = {
                    'sed': batch_sample['sed_label'],
                    'doa': batch_sample['doa_label'],
                }
                if self.cuda:
                    batch_x = batch_x.to(self.rank, non_blocking=True)
                    batch_target['sed'] = batch_target['sed'].to(self.rank, non_blocking=True)
                    batch_target['doa'] = batch_target['doa'].to(self.rank, non_blocking=True)

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
                loss_dict = self.losses.calculate(pred, batch_target, epoch_it)

                pred['sed'] = torch.sigmoid(pred['sed'])
                loss_all += loss_dict['all'].cpu().detach().numpy()
                loss_sed += loss_dict['sed'].cpu().detach().numpy()
                loss_doa += loss_dict['doa'].cpu().detach().numpy()
                pred_sed_list.append(pred['sed'].cpu().detach().numpy())
                pred_doa_list.append(pred['doa'].cpu().detach().numpy())

            pred_sed = np.concatenate(pred_sed_list, axis=0)
            pred_doa = np.concatenate(pred_doa_list, axis=0)

            origin_num_clips = int(pred_sed.shape[0]/self.num_segments)
            origin_T = int(pred_sed.shape[1]*self.num_segments)
            pred_sed = pred_sed.reshape((origin_num_clips, origin_T, 3, -1))[:, :int(self.clip_length / self.label_resolution)]
            pred_doa = pred_doa.reshape((origin_num_clips, origin_T, 3, -1))[:, :int(self.clip_length / self.label_resolution)]

            pred_sed_max = pred_sed.max(axis=-1)
            pred_sed_max_idx = pred_sed.argmax(axis=-1)
            pred_sed = np.zeros_like(pred_sed)
            for b_idx in range(origin_num_clips):
                for t_idx in range(origin_T):
                    for track_idx in range(3):
                        pred_sed[b_idx, t_idx, track_idx, pred_sed_max_idx[b_idx, t_idx, track_idx]] = \
                            pred_sed_max[b_idx, t_idx, track_idx]
            pred_sed = (pred_sed > self.cfg['training']['threshold_sed']).astype(np.float32)

            pred_list = track_to_list(pred_sed, pred_doa, self.max_loc_value)
            true_list = self.dev_gt_seld_labels
            
            out_losses = {
                'loss_all': loss_all / (batch_idx + 1),
                'loss_sed': loss_sed / (batch_idx + 1),
                'loss_doa': loss_doa / (batch_idx + 1),
            }
            for k, v in out_losses.items():
                out_losses[k] = reduce_value(v)
            metrics = self.metrics.compute_global_metrics(pred_list, true_list)
            metrics['TP'] = reduce_value(metrics['TP'], average=False) 
            metrics['FP'] = reduce_value(metrics['FP'], average=False) 
            metrics['FN'] = reduce_value(metrics['FN'], average=False) 
            metrics_scores = self.metrics.compute_score(metrics)

            return out_losses, metrics_scores

