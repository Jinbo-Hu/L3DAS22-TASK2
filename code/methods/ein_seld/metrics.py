import pandas as pd
import numpy as np
import os
import sys

class Metrics(object):
    """Metrics for evaluation

    """
    def __init__(self, dataset):

        self.metrics = []
        self.names = ['F score','precision','recall']

        self.num_classes = dataset.num_classes
        self.spatial_threshold = 1 # 1 meter
        self.num_frames = dataset.clip_length / dataset.label_resolution
        self.max_ov = dataset.max_ov
        self.label_dic_task2 = dataset.label_dic_task2

    def compute_global_metrics(self, pred, true):
        '''
        Compute TP, FP, FN of global data using
        location sensitive detection

        :param pred: list containing predicted sound event. Each row: [[event1, x, y, z],[event2, x, y, z],[event3, x, y, z]]
        :param true: list containing true sound event. Each row: [[event1, x, y, z],[event2, x, y, z],[event3, x, y, z]]
        '''
        all_frames = len(pred)
        TP = 0   #true positives
        FP = 0   #false positives
        FN = 0   #false negatives
        for frame in range(all_frames):
            t = true[frame]
            p = pred[frame]
            matched = 0           #counts the matching events
            match_ids_p = []       #all pred ids that matched
            match_ids_t = []     #all truth ids that matched

            if len(t) == 0:         #if there are PREDICTED but not TRUE events
                FP += len(p)        #all predicted are false positive
            elif len(p) == 0:       #if there are TRUE but not PREDICTED events
                FN += len(t)        #all predicted are false negative
            else:
                for event_t in range(len(t)):           #iterate all true events
                    #count if in each true event there is or not a matching predicted event
                    true_class = t[event_t][0]          #true class
                    true_coord = np.array(t[event_t][-3:])      #true coordinates
                    for event_p in range(len(p)):       #compare each true event with all predicted events
                        pred_class = p[event_p][0]      #predicted class
                        pred_coord = np.array(p[event_p][-3:])    #predicted coordinates
                        spat_error = np.linalg.norm(true_coord-pred_coord)  #cartesian distance between spatial coords
                        #if predicton is correct (same label + not exceeding spatial error threshold)
                        if true_class == pred_class and spat_error < self.spatial_threshold:  
                            match_ids_t.append(event_t)
                            match_ids_p.append(event_p)
                            
                unique_ids_p = np.unique(match_ids_p)  #remove duplicates from matches ids lists
                unique_ids_t = np.unique(match_ids_t)  #remove duplicates from matches ids lists
                matched = min(len(unique_ids_p), len(unique_ids_t))   #compute the number of actual matches without duplicates

                num_true_items = len(t)
                num_pred_items = len(p)
                fn =  num_true_items - matched
                fp = num_pred_items - matched
                #add to counts
                TP += matched          #number of matches are directly true positives
                FN += fn
                FP += fp               

        results = {
            'TP': TP,
            'FP': FP,
            'FN': FN
            }
        return results

    def compute_score(self, metrics):
        '''
        Compute F1 score from metrics based on the
        location sensitive detection metric
        '''
        TP = metrics['TP']
        FP = metrics['FP']
        FN = metrics['FN']
        precision = TP / (TP + FP + sys.float_info.epsilon)
        recall = TP / (TP + FN + sys.float_info.epsilon)
        F_score = 2 * ((precision * recall) / (precision + recall + sys.float_info.epsilon))
        score = {
            'F score': F_score,
            'precision': precision,
            'recall': recall,
        }
        return score
                

