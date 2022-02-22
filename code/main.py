import sys
from timeit import default_timer as timer

import torch

from learning import (evaluate, infer, initialize, preprocess,
                      salsa_feature_extraction, train)
from utils.cli_parser import parse_cli_overides
from utils.config import get_dataset
from utils.ddp_init import cleanup, run, setup
from utils.common import prepare_train_id


def training(rank, args, cfg, dataset):
    # Init DDP
    setup(rank=rank, world_size=torch.cuda.device_count(),args=args)
    train_initializer = initialize.init_train(args, cfg, dataset)
    train.train(cfg, **train_initializer)


def main(args, cfg):
    """Execute a task based on the given command-line arguments.
    
    This function is the main entry-point of the program. It allows the
    user to extract features, train a model, infer predictions, and
    evaluate predictions using the command-line interface.

    Args:
        args: command line arguments.
        cfg: configurations.
    Return:
        0: successful termination
        'any nonzero value': abnormal termination
    """

    begin_time = timer()
    
    # Dataset initialization
    dataset = get_dataset(root_dir=cfg['dataset_dir'], cfg=cfg)

    # Preprocess
    if args.mode == 'preprocess':
        preprocessor = preprocess.Preprocessor_task2(args, cfg, dataset)
        if args.preproc_mode == 'extract_data':
            preprocessor.extract_data()
        if args.preproc_mode == 'extract_frame_label':
            preprocessor.extract_frame_label()
        if args.preproc_mode == 'extract_track_label':
            preprocessor.extract_track_label()
        if args.preproc_mode == 'extract_scalar':
            preprocessor.extract_scalar()
        if args.preproc_mode == 'salsa_extractor':
            salsa_feature_extraction.extract_features(args, cfg)
    
    # Train
    if args.mode == 'train':
        prepare_train_id(args, cfg)
        run(training, args, cfg, dataset)

    # Inference
    elif args.mode == 'infer':
        infer_initializer = initialize.init_infer(args, cfg, dataset)
        infer.infer(cfg, dataset, **infer_initializer)

    # Evaluate
    elif args.mode == 'evaluate':
        evaluate.evaluate(cfg, dataset)
    
    # cleanup()
    print('The total time:',timer()-begin_time)
    return 0


if __name__ == '__main__':
    args, cfg = parse_cli_overides()
    sys.exit(main(args, cfg))
