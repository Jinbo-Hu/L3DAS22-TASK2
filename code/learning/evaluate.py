from pathlib import Path

import numpy as np
import pandas as pd
from utils.common import csv_to_list
from methods.ein_seld.metrics import Metrics

def evaluate(cfg, dataset):

    """ Evaluate scores

    """

    metrics = Metrics(dataset)

    '''Directories'''
    print('Inference ID is {}\n'.format(cfg['inference']['infer_id']))

    out_infer_dir = Path(cfg['workspace_dir']).joinpath('results').joinpath('out_infer')\
        .joinpath(cfg['method']).joinpath(cfg['inference']['infer_id'])
    submissions_dir = out_infer_dir.joinpath('submissions')

    true_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('meta').joinpath('frame')
    pred_dir = submissions_dir

    paths_list = [path for path in sorted(pred_dir.glob('*.csv')) if not path.name.startswith('.')]
 
    if cfg['inference']['testset_type'] == 'dev':
        true_dir = true_dir.joinpath('dev')
    elif cfg['inference']['testset_type'] == 'test':
        true_dir = true_dir.joinpath('eval')
    elif cfg['inference']['testset_type'] == 'train':
        true_dir = true_dir.joinpath('train')
    
    pred_all = []
    true_all = []
    for pred_path in paths_list:
        fn = pred_path.stem
        true_path = true_dir.joinpath('label_' + fn + '.csv')
        pred = csv_to_list(pred_path, dataset)
        true = csv_to_list(true_path, dataset)
        pred_all.extend(pred)
        true_all.extend(true)
    measure = metrics.compute_global_metrics(pred_all, true_all)
    metrics_scores = metrics.compute_score(measure)
    


    out_str = 'test: '
    for key, value in metrics_scores.items():
        out_str += '{}: {:.3f},  '.format(key, value)
    print('---------------------------------------------------------------------------------------------------'
        +'-------------------------------------------------')
    print(out_str)
    print('---------------------------------------------------------------------------------------------------'
        +'-------------------------------------------------')

    out_eval_dir = Path(cfg['workspace_dir']).joinpath('results').joinpath('out_eval')\
        .joinpath(cfg['method']).joinpath(cfg['inference']['infer_id'])
    out_eval_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_eval_dir.joinpath('results.csv')
    df = pd.DataFrame(metrics_scores, index=[0])
    df.to_csv(result_path, sep=',', mode='a')

