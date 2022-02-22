"""
This module extract salsa features for first order ambisonics formats.
Reference: https://github1s.com/thomeou/SALSA/blob/HEAD/dataset/salsa_feature_extraction.py
The MIT License
"""
import os
import shutil

import h5py
import librosa
import numpy as np
from sklearn import preprocessing
from timeit import default_timer as timer
from tqdm import tqdm
import sys
from pathlib import Path

from utils.datasets import l3das21, l3das22
dataset_dict = {
    'l3das22': l3das22,
    'l3das21': l3das21
}
def extract_normalized_eigenvector(X, condition_number: float = 5.0, n_hopframes: int = 3, is_tracking: bool = True,
                                    fs: int = None, n_fft: int = None, lower_bin: int = None):
    """
    Function to extract normalized eigenvector.
    :param X: <np.ndarray (n_bins, n_frames, n_chans): Clipped spectrogram between lower_bin and upper_bin.
    :param is_tracking: If True, use noise-floor tracking
    :param audio_format: Choice: 'foa' (take real part)| 'mic' (take phase)
    """
    # get size of X
    n_bins, n_frames, n_chans = X.shape

    # noise floor tracking params
    n_sig_frames = 3
    indicator_countdown = n_sig_frames * np.ones((n_bins,), dtype=int)
    alpha = 0.02
    slow_scale = 0.1
    floor_up = 1 + alpha
    floor_up_slow = 1 + slow_scale * alpha
    floor_down = 1 - alpha
    snr_ratio = 1.5


    # padding X for Rxx computation
    X = np.pad(X, ((0, 0), (n_hopframes, n_hopframes), (0, 0)), 'wrap')

    # select type of signal for bg noise tracking:
    ismag = 0  # 0 use running average for tracking, 1 use raw magnitude for tracking
    signal_magspec = np.zeros((n_bins, n_frames))
    # signal to track
    n_autocorr_frames = 3
    if ismag == 1:
        signal_magspec = np.abs(X[:, n_hopframes:n_hopframes + n_frames, 0])
    else:
        for iframe in np.arange(n_autocorr_frames):
            signal_magspec = signal_magspec + np.abs(X[:, n_hopframes - iframe:n_hopframes - iframe + n_frames, 0]) ** 2
        signal_magspec = np.sqrt(signal_magspec / n_autocorr_frames)

    # Initial noisefloor assuming first few frames are noise
    noise_floor = 0.5*np.mean(signal_magspec[:, 0:5], axis=1)

    # memory to store output
    normalized_eigenvector_mat = np.zeros((n_chans - 1, n_bins, n_frames))  # normalized eigenvector of ss tf bin
    # =========================================================================
    for iframe in np.arange(n_hopframes, n_frames + n_hopframes):
        # get current frame tracking singal
        xfmag = signal_magspec[:, iframe - n_hopframes]
        # ---------------------------------------------------------------------
        # bg noise tracking: implement direct up/down noise floor tracker
        above_noise_idx = xfmag > noise_floor
        # ------------------------------------
        # if signal above noise floor
        indicator_countdown[above_noise_idx] = indicator_countdown[above_noise_idx] - 1
        negative_indicator_idx = indicator_countdown < 0
        # update noise slow for bin above noise and negative indicator
        an_ni_idx = np.logical_and(above_noise_idx, negative_indicator_idx)
        noise_floor[an_ni_idx] = floor_up_slow * noise_floor[an_ni_idx]
        # update noise for bin above noise and positive indicator
        an_pi_idx = np.logical_and(above_noise_idx, np.logical_not(negative_indicator_idx))
        noise_floor[an_pi_idx] = floor_up * noise_floor[an_pi_idx]
        # reset indicator counter for bin below noise floor
        indicator_countdown[np.logical_not(above_noise_idx)] = n_sig_frames
        # reduce noise floor for bin below noise floor
        noise_floor[np.logical_not(above_noise_idx)] = floor_down * noise_floor[np.logical_not(above_noise_idx)]
        # make sure noise floor does not go to 0
        noise_floor[noise_floor < 1e-6] = 1e-6
        # --------------------------------------
        # select TF bins above noise level
        indicator_sig = xfmag > (snr_ratio * noise_floor)
        # ---------------------------------------------------------------------
        # valid bin after onset and noise background tracking
        if is_tracking:
            valid_bin = indicator_sig
        else:
            valid_bin = np.ones((n_bins,), dtype='bool')
        # ---------------------------------------------------------------------
        # coherence test
        for ibin in np.arange(n_bins):
            if valid_bin[ibin]:
                # compute covariance matrix using (2*nframehop + 1) frames
                X1 = X[ibin, iframe - n_hopframes:iframe + n_hopframes + 1, :]  # (2*n_hopframes+1) x nchan
                Rxx1 = np.dot(X1.T, X1.conj()) / float(2 * n_hopframes + 1)

                # svd: u: n_chans x n_chans, s: n_chans, columns of u is the singular vectors
                u, s, v = np.linalg.svd(Rxx1)

                # coherence test
                if s[0] > s[1] * condition_number:
                    indicator_rank1 = True
                else:
                    indicator_rank1 = False
                # update valid bin
                if is_tracking:
                    valid_bin[ibin] = valid_bin[ibin] and indicator_rank1

                # compute doa spectrum
                if valid_bin[ibin]:
                    # normalize largest eigenvector
                    normed_eigenvector = np.real(u[1:, 0] / (u[0, 0] + sys.float_info.epsilon))
                    normed_eigenvector = normed_eigenvector/ (np.sqrt(np.sum(normed_eigenvector**2)) + sys.float_info.epsilon)
                    # save output
                    normalized_eigenvector_mat[:, ibin, iframe - n_hopframes] = normed_eigenvector

    return normalized_eigenvector_mat


class MagStftExtractor:
    """
    Extract single-channel or multi-channel log-linear spectrograms. return feature of shape 4 x n_timesteps x 200
    """
    def __init__(self, n_fft: int, hop_length: int, win_length: int = None, window: str = 'hann',
                 is_compress_high_freq: bool = True):
        """
        :param n_fft: Number of FFT points.
        :param hop_length: Number of sample for hopping.
        :param win_length: Window length <= n_fft. If None, assign n_fft
        :param window: Type of window.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        if win_length is None:
            self.win_length = self.n_fft
        else:
            self.win_length = win_length
        assert self.win_length <= self.n_fft, 'Windown length is greater than nfft!'
        assert n_fft == 512 or n_fft == 256, 'nfft is not 512 or 256'
        if is_compress_high_freq:
            if n_fft == 512:
                self.W = np.zeros((200, 257), dtype=np.float32)
                for i in np.arange(192):
                    self.W[i, i+1] = 1.0
                for i in np.arange(192, 200):
                    if i < 199:
                        self.W[i, 193 + (i-192) * 8: 193 + (i-192) * 8 + 8] = 1/8
                    elif i == 199:
                        self.W[i, 193 + (i-192) * 8: 193 + (i-192) * 8 + 7] = 1/8
            elif n_fft == 256:
                self.W = np.zeros((100, 129), dtype=np.float32)
                for i in np.arange(96):
                    self.W[i, i+1] = 1.0
                for i in np.arange(96, 100):
                    if i < 99:
                        self.W[i, 97 + (i-96) * 8: 97 + (i-96) * 8 + 8] = 1/8
                    elif i == 99:
                        self.W[i, 97 + (i-96) * 8: 97 + (i-96) * 8 + 7] = 1/8
        else:
            self.W = np.zeros((n_fft // 2, n_fft // 2 + 1), dtype=np.float32)
            for i in np.arange(n_fft // 2):
                self.W[i, i + 1] = 1.0

    def extract(self, audio_input: np.ndarray) -> np.ndarray:
        """
        :param audio_input: <np.ndarray: (n_channels, n_samples)>.
        :return: logmel_features <np.ndarray: (n_channels, n_timeframes, n_features)>.
        """
        n_channels = audio_input.shape[0]
        log_features = []

        for i_channel in range(n_channels):
            spec = np.abs(librosa.stft(y=np.asfortranarray(audio_input[i_channel]),
                                       n_fft=self.n_fft,
                                       hop_length=self.hop_length,
                                       win_length=self.win_length,
                                       center=True,
                                       window=self.window,
                                       pad_mode='reflect'))

            spec = np.dot(self.W, spec**2).T
            log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
            log_spec = np.expand_dims(log_spec, axis=0)
            log_features.append(log_spec)

        log_features = np.concatenate(log_features, axis=0)

        return log_features


def compute_scaler(feature_dir: str) -> None:
    """
    Compute feature mean and std vectors of spectrograms for normalization.
    :param feature_dir: Feature directory that contains train and test folder.
    """
    print('============> Start calculating scaler')
    start_time = timer()

    # Get list of feature filenames
    train_feature_dir = Path(os.path.join(feature_dir))
    dataset_type = ['dev', 'train']
    feature_fn_list = []
    for type in dataset_type:
        feature_fn_list.extend([path for path in sorted(train_feature_dir.joinpath(type).glob('*.h5')) \
                    if not path.name.startswith('.')])

    # Get the dimensions of feature by reading one feature files
    full_feature_fn = feature_fn_list[0]
    with h5py.File(full_feature_fn, 'r') as hf:
        afeature = hf['feature'][:]  # (n_chanels, n_timesteps, n_features)
    n_channels = afeature.shape[0]
    assert n_channels == 14, 'only support n_channels = 14, got {}'.format(n_channels)
    n_feature_channels = [0,1,2,3,7,8,9,10]  # hard coded number

    # initialize scaler
    scaler_dict = {}
    for i_chan in np.arange(len(n_feature_channels)):
        scaler_dict[i_chan] = preprocessing.StandardScaler()

    # Iterate through data
    for count, feature_fn in enumerate(tqdm(feature_fn_list)):
        full_feature_fn = feature_fn
        with h5py.File(full_feature_fn, 'r') as hf:
            afeature = hf['feature'][:]  # (n_chanels, n_timesteps, n_features)
            for i_chan in range(len(n_feature_channels)):
                scaler_dict[i_chan].partial_fit(afeature[n_feature_channels[i_chan], :, :])  # (n_timesteps, n_features)


    # Extract mean and std
    feature_mean = []
    feature_std = []
    for i_chan in range(len(n_feature_channels)):
        feature_mean.append(scaler_dict[i_chan].mean_)
        feature_std.append(np.sqrt(scaler_dict[i_chan].var_))

    feature_mean = np.array(feature_mean)
    feature_std = np.array(feature_std)

    # Expand dims for timesteps: (n_chanels, n_timesteps, n_features)
    feature_mean = np.expand_dims(feature_mean, axis=1)
    feature_std = np.expand_dims(feature_std, axis=1)

    scaler_path = os.path.join(feature_dir + 'feature_scaler.h5')
    with h5py.File(scaler_path, 'w') as hf:
        hf.create_dataset('mean', data=feature_mean, dtype=np.float32)
        hf.create_dataset('std', data=feature_std, dtype=np.float32)

    print('Features shape: {}'.format(afeature.shape))
    print('mean {}: {}'.format(feature_mean.shape, feature_mean))
    print('std {}: {}'.format(feature_std.shape, feature_std))
    print('Scaler path: {}'.format(scaler_path))
    print('Elapsed time: {:.3f} s'.format(timer() - start_time))


def extract_features(args: dict,
                     cfg: dict,
                     cond_num: float = 5,  # 5, 0
                     n_hopframes: int = 3,   # do not change
                     is_tracking: bool = True,  # Better to do tracking
                     is_compress_high_freq: bool = False,
                     task: str = 'feature_scaler') -> None:
    """
    Extract salsa features: log-linear spectrogram + normalized eigenvector (magnitude for FOA, phase for MIC)
    :param args: command line arguments.
    :param cfg: configurations.
    :param data_config: Path to data config file.
    :param cond_num: threshold for ddr for coherence test.
    :param n_hopframes: Number of adjacent frames to compute covariance matrix.
    :param is_tracking: If True, do noise-floor tracking.
    :param is_compress_high_freq: If True, compress high frequency region to reduce feature dimension.
    :param task: 'feature_scaler': extract feature and scaler, 'feature': only extract feature, 'scaler': only extract
        scaler.
    """

    feature_type = 'tfmap'

    dataset = dataset_dict[cfg['dataset']](cfg['dataset_dir'], cfg)

    # Parse config file
    cfg_SALSA = cfg['data']['SALSA']
    fs = cfg_SALSA['sample_rate']
    n_fft = cfg_SALSA['n_fft']
    hop_length = cfg_SALSA['hop_length']
    win_length = cfg_SALSA['win_length']

    # Doa info
    n_mics = 4
    fmin = cfg_SALSA['fmin']
    fmax = cfg_SALSA['fmax']
    fmax = np.min((fmax, fs // 2))
    n_bins = n_fft // 2 + 1
    lower_bin = int(np.floor(fmin * n_fft / float(fs)))  # 512: 1; 256: 0
    upper_bin = int(np.floor(fmax * n_fft / float(fs)))  # 9000Hz: 512: 192, 256: 96
    lower_bin = np.max((1, lower_bin))

    assert n_fft == 512 or n_fft == 256, 'only 256 or 512 fft is supported'
    if is_compress_high_freq:
        if n_fft == 512:
            freq_dim = 200
        elif n_fft == 256:
            freq_dim = 100
    else:
        freq_dim = n_fft // 2

    # Get feature descriptions
    feature_description = '{}fs_{}nfft_{}nhop_{}cond_{}fmaxdoa'.format(
        fs, n_fft, hop_length, int(cond_num), int(fmax))
    if not is_tracking:
        feature_description = feature_description + '_notracking'
    if not is_compress_high_freq:
        feature_description = feature_description + '_nocompress'

    # Get feature extractor
    stft_feature_extractor = MagStftExtractor(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                              is_compress_high_freq=is_compress_high_freq)

    print('Feature description: {}'.format(feature_description))
    ambisonics_format = ['A','B']
    if args.dataset_type == 'train':
        dataset_type = ['dev','train']
    if args.dataset_type == 'test':
        dataset_type = ['test']
    # Extract features
    if task in ['feature_scaler', 'feature']:
        for type in dataset_type:
            print('============> Start extracting features for {} split'.format(type))
            start_time = timer()
            # Required directories
            audio_dir = os.path.join(dataset.dataset_dir['task2'][type], 'data')
            main_feature_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('task2').joinpath('features')
            feature_dir = os.path.join(main_feature_dir, feature_type, feature_description, type)
            audio_dir = Path(audio_dir)
            feature_dir = Path(feature_dir)
            # Empty feature folder
            shutil.rmtree(feature_dir, ignore_errors=True)
            feature_dir.mkdir(parents=True, exist_ok=True)

            # Get audio list
            audio_fn_list = [fn for fn in sorted(os.listdir(audio_dir)) if fn.split('.')[0].split('_')[-1]=='A']
            # Extract features
            for count, audio_fn in enumerate(tqdm(audio_fn_list)):
                features = []
                for format in ambisonics_format:
                    audio_fn = audio_fn.replace('_A', '_'+format)
                    full_audio_fn = os.path.join(audio_dir, audio_fn)
                    audio_input, _ = librosa.load(full_audio_fn, sr=fs, mono=False, dtype=np.float32)
                    # Extract stft feature (already remove the first frequency bin, correspond to fmin)
                    stft_feature = stft_feature_extractor.extract(audio_input)  # (n_channels, n_timesteps, 200)

                    # Extract mask and doa
                    # Compute stft
                    for imic in np.arange(n_mics):
                        stft = librosa.stft(y=np.asfortranarray(audio_input[imic, :]), n_fft=n_fft, hop_length=hop_length,
                                            center=True, window='hann', pad_mode='reflect')
                        if imic == 0:
                            n_frames = stft.shape[1]
                            afeature = np.zeros((n_bins, n_frames, n_mics), dtype='complex')
                        afeature[:, :, imic] = stft
                    X = afeature[lower_bin:upper_bin, :, :]
                    # compute normalized eigenvector
                    normed_eigenvector_mat = extract_normalized_eigenvector(
                        X, condition_number=cond_num, n_hopframes=n_hopframes, is_tracking=is_tracking,
                        fs=fs, n_fft=n_fft, lower_bin=lower_bin,)

                    # lower_bin now start at 0
                    full_eigenvector_mat = np.zeros((n_mics - 1, n_frames, freq_dim))
                    full_eigenvector_mat[:, :, :(upper_bin - lower_bin)] = np.transpose(normed_eigenvector_mat, (0, 2, 1))

                    # Stack features
                    audio_feature = np.concatenate((stft_feature, full_eigenvector_mat), axis=0)
                    features.append(audio_feature)
                features = np.concatenate(features, axis=0)
                # Write features to file
                audio_fn = audio_fn.replace('_B','')
                feature_fn = os.path.join(feature_dir, audio_fn.replace('wav', 'h5'))
                with h5py.File(feature_fn, 'w') as hf:
                    hf.create_dataset('feature', data=features, dtype=np.float32)
                tqdm.write('{}, {}, {}'.format(count, audio_fn, features.shape))
            print("Extracting feature finished! Elapsed time: {:.3f} s".format(timer() - start_time))

    # Compute feature mean and std for train set. For simplification, we use same mean and std for validation and
    # evaluation
    if task in ['feature_scaler', 'scaler'] and args.dataset_type == 'train':
        feature_dir = os.path.join(main_feature_dir, feature_type, feature_description)
        compute_scaler(feature_dir=feature_dir)


