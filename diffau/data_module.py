
from os.path import join
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
import pandas as pd
from functools import lru_cache
from scipy.io import loadmat
import grids
from diffau.util.other import create_sh_matrix, quick_convolve, rotate_sh


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames, output_order, first_low_order_channel,
            normalize=True, spec_transform=None, stft_kwargs=None, 
            apply_rotation=False, rotation_prob=0.75, rotation_range_azi=(0, 360), 
            rotation_range_theta=(0, 180), data_multiplier=1, **ignored_kwargs):


        # Read file paths according to file naming format.

        self.subset = subset
        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.output_order = output_order
        self.first_low_order_channel = first_low_order_channel
        
        # Rotation augmentation parameters
        self.apply_rotation = apply_rotation
        self.rotation_prob = rotation_prob
        self.rotation_range_azi = rotation_range_azi
        self.rotation_range_theta = rotation_range_theta
        self.data_multiplier = data_multiplier

        # Apply data multiplier: replicate file list
        
        if dummy !=-1 and subset == 'validation' and dummy<100:
            #in small dummy, do eval on train overfit (to see the full reverse metrics)
            self.files = sorted(glob(join(data_dir, 'train', "*.wav")))
            self.meta_data = pd.read_csv(glob(join(data_dir,"*.tsv"))[0],sep='\t').query("type == 'train'")
        else:
            self.files = sorted(glob(join(data_dir, subset, "*.wav")))
            self.meta_data = pd.read_csv(glob(join(data_dir,"*.tsv"))[0],sep='\t').query("type == @subset")

        if self.data_multiplier > 1:
            self.files = self.files * self.data_multiplier
            print(f"{subset} : Applied data multiplier of {self.data_multiplier}x: {len(self.files)} samples total",flush=True)
        print(f" {subset} Apply Rotation set to {self.apply_rotation} | Azi {rotation_range_azi} , Theta {rotation_range_theta}", flush=True)

        print(f"Found {len(self.files)} in {join(data_dir, subset)}")

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        x, _ = load(self.files[i])
        # formula applies for center=True
        x = x[self.first_low_order_channel:(self.output_order+1)**2] #Remove irrelvant channels
        
        # Apply rotation augmentation (only for training set)
        if self.subset == 'train' and self.apply_rotation:
            # Apply rotation with specified probability
            if np.random.rand() < self.rotation_prob:
                # Sample random rotation angles
                azimuth = np.random.uniform(self.rotation_range_azi[0], self.rotation_range_azi[1])
                theta = np.random.uniform(self.rotation_range_theta[0], self.rotation_range_theta[1])
                
                # rotate_sh expects [n_samples, n_channels] format
                x_rotated = rotate_sh(x.T, azimuth_deg=azimuth, elevation_deg=theta, sh_order=self.output_order)
                # Transpose back to [n_channels, n_samples]
                x = x_rotated.T
        
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        assert pad < self.stft_kwargs['n_fft'], f"Padding is {pad}, which is larger than 1 fft {self.stft_kwargs['n_fft']} - blank windows!"

        # Assumes the audio is a bit smaller than needed num of frames
        # pad audio if the length T is smaller than num_frames
        x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')

        # normalize w.r.t to the noisy or the clean signal or not at all
        if self.normalize:
            normfac = x.abs().max()
        else:
            normfac = 1.0
        x = x / normfac

        X = torch.stft(x, **self.stft_kwargs)
        X = self.spec_transform(X)
        return X

    def __len__(self):
        return len(self.files) if self.dummy==-1  else min(self.dummy,len(self.files))


class AmbisonicSpecDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, default = None, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories.")
        parser.add_argument("--format", type=str, choices=("default", "reverb"), default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=4, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", type=int, default=-1, help="size of train set, if -1 then uses all of it")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=int, choices=(1,0), default=1, help="Normalize the input waveforms 1 is true 0 false.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        parser.add_argument("--apply_rotation", type=int, choices=(1,0), default=0, help="Apply random rotation augmentation to training data. 1 is true 0 false.")
        parser.add_argument("--rotation_prob", type=float, default=0.75, help="Probability of applying rotation to each sample. 0.75 by default.")
        parser.add_argument("--data_multiplier", type=int, default=1, help="Dataset size multiplier for augmentation. Each sample will be reused this many times with different rotations.")
        return parser

    def __init__(
        self, base_dir, format='default', batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=-1, spec_factor=0.15, spec_abs_exponent=0.5, output_order = 10,first_low_order_channel = 0,
        gpu=True, normalize=True, transform_type="exponent", apply_rotation=False, rotation_prob=0.75,
        data_multiplier=1, **kwargs):

        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.output_order = output_order
        self.first_low_order_channel = first_low_order_channel
        self.apply_rotation = apply_rotation #only passed to train
        self.rotation_prob = rotation_prob #only passed to train
        self.data_multiplier = data_multiplier #only passed to train
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format, output_order=self.output_order,
                first_low_order_channel=self.first_low_order_channel, normalize=self.normalize, data_multiplier=self.data_multiplier,
                apply_rotation=self.apply_rotation, rotation_prob=self.rotation_prob, **specs_kwargs)
            self.valid_set = Specs(data_dir=self.base_dir, subset='validation',
                dummy=self.dummy, shuffle_spec=False, format=self.format, output_order=self.output_order,
                first_low_order_channel=self.first_low_order_channel,normalize=self.normalize, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False, format=self.format, output_order=self.output_order,
                first_low_order_channel=self.first_low_order_channel, normalize=self.normalize, **specs_kwargs)
            
        self.meta_data = pd.read_csv(glob(join(self.base_dir, '*.tsv'))[0], sep='\t')

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        """
        perform istft on [B,C,F,T] or [C,F,T] tensor
        """
        window = self._get_window(spec)
        added_batch_dim = False
        if spec.dim() == 3:
            spec = spec.unsqueeze(0)  # add batch dim
        B,C,F,T = spec.shape
        spec = spec.view(B*C,F,T)
        waveform = torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length}).view(B,C,-1)
        if added_batch_dim:
            waveform = waveform.squeeze(0)
        return waveform

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers,persistent_workers=True if self.num_workers else False, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers,persistent_workers=True if self.num_workers else False, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers,persistent_workers=True if self.num_workers else False, pin_memory=self.gpu, shuffle=False
        )
