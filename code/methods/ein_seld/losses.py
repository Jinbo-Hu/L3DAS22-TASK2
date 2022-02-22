import numpy as np
import torch
import sys
from methods.utils.loss_utilities import BCEWithLogitsLoss, MSELoss


class Losses:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.beta = cfg['training']['loss_beta']
        self.losses = [BCEWithLogitsLoss(reduction='mean'), MSELoss(reduction='mean')]
        self.losses_pit = [BCEWithLogitsLoss(reduction='PIT'), MSELoss(reduction='PIT')]
        self.names = ['loss_all'] + [loss.name for loss in self.losses] 

    def calculate(self, pred, target, epoch_it=0):

        if 'PIT' not in self.cfg['training']['PIT_type']:
            loss_sed = self.losses[0].calculate_loss(pred['sed'], target['sed'])
            loss_doa = self.losses[1].calculate_loss(pred['doa'], target['doa'])
        elif self.cfg['training']['PIT_type'] == 'tPIT':
            loss_sed, loss_doa = self.tPIT(pred, target)
        loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa
        losses_dict = {
            'all': loss_all,
            'sed': loss_sed,
            'doa': loss_doa,
        }
        return losses_dict    
    

    #### modify tracks
    def tPIT(self, pred, target):
        """Frame Permutation Invariant Training for 6 possible combinations

        Args:
            pred: {
                'sed': [batch_size, T, num_tracks=3, num_classes], 
                'doa': [batch_size, T, num_tracks=3, doas=3]
            }
            target: {
                'sed': [batch_size, T, num_tracks=3, num_classes], 
                'doa': [batch_size, T, num_tracks=3, doas=3]            
            }
        Return:
            loss_sed: Find a possible permutation to get the lowest loss of sed. 
            loss_doa: Find a possible permutation to get the lowest loss of doa. 
        """


        target1 = {
            'sed': target['sed'][:,:,[0,2,1],:],
            'doa': target['doa'][:,:,[0,2,1],:]
        }
        target2 = {
            'sed': target['sed'][:,:,[1,2,0],:],
            'doa': target['doa'][:,:,[1,2,0],:]
        }
        target3 = {
            'sed': target['sed'][:,:,[1,0,2],:],
            'doa': target['doa'][:,:,[1,0,2],:]
        }
        target4 = {
            'sed': target['sed'][:,:,[2,0,1],:],
            'doa': target['doa'][:,:,[2,0,1],:]
        }
        target5 = {
            'sed': target['sed'][:,:,[2,1,0],:],
            'doa': target['doa'][:,:,[2,1,0],:]
        }
        target_flip = [target, target1, target2, target3, target4, target5]
        
        loss_sed_list = []
        loss_doa_list = []
        loss_ = []
        loss_sed = 0
        loss_doa = 0
        for i in range(6):
            loss_sed_list.append(self.losses_pit[0].calculate_loss(pred['sed'], target_flip[i]['sed'])) 
            loss_doa_list.append(self.losses_pit[1].calculate_loss(pred['doa'], target_flip[i]['doa']))
            loss_.append(loss_sed_list[i]+loss_doa_list[i])
        for i in range(6):
            min = 1
            for j in range(6):
                if i != j:
                    min = min * (loss_[i]<loss_[j])
            loss_sed = loss_sed + min * loss_sed_list[i]
            loss_doa = loss_doa + min * loss_doa_list[i]
        loss_sed = loss_sed + loss_sed_list[0] * (loss_[0]==loss_[1]) *  (loss_[0]==loss_[2]) * (loss_[0]==loss_[3]) *\
            (loss_[0]==loss_[4]) *(loss_[0]==loss_[5]) 
        loss_doa = loss_doa + loss_doa_list[0] * (loss_[0]==loss_[1]) *  (loss_[0]==loss_[2]) * (loss_[0]==loss_[3]) *\
            (loss_[0]==loss_[4]) *(loss_[0]==loss_[5]) 
        loss_sed = loss_sed.mean()
        loss_doa = loss_doa.mean()
        
        return loss_sed, loss_doa




