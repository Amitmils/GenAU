import torch
import os
import spaudiopy as spa
import math
import pandas as pd
import torchaudio
import sounddevice as sd
import soundfile as sf
import random
from typing import Optional
from torch.nn import functional as F
from scipy.io import loadmat
from diffau.util.other import create_sh_matrix, complex_to_real_ambisonics, real_to_complex_ambisonics,quick_convolve,sh_freq_complement,rotate_sh
import grids
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
import torchaudio


def trim_zero_columns_from_right(C):
    # RIR: of shape ((N+1)**2, T)
    nonzero_cols = torch.any(C != 0, dim=0)  # shape (T,)
    if not torch.any(nonzero_cols):
        return C[:, :0]  # All columns are zero
    last_nonzero_idx = torch.where(nonzero_cols)[0][-1]
    return C[:, :last_nonzero_idx + 1]

def apply_random_gain(signal: torch.Tensor, min_db: float = -6, max_db: float = 6):
    """
    Apply random gain in dB range to signal.
    
    Args:
        signal: Input audio signal tensor
        min_db: Minimum gain in dB (default: -6)
        max_db: Maximum gain in dB (default: +6)
    
    Returns:
        Tuple of (gained_signal, gain_db)
    """
    gain_db = random.uniform(min_db, max_db)
    gain_linear = 10 ** (gain_db / 20)
    return signal * gain_linear, gain_db

def encode_signal(
    s : torch.tensor,
    dry_fs : int,
    sh_order : int = 1,
    ph : Optional[float] = None,
    th : Optional[float] = None,
    dir_type  : Literal["deg","rad"] = "rad",
    encoding_type : Literal["real","complex"] = "real",
    signal_domain  : Literal["time","freq"]= "time",
    n_fft : int  = 1024,        # Number of FFT points
    plot  = False,
    rir_path : Optional[str] = None,
    hoa_t_cutoff :int = -1,
    phi_rotation : float = 0.0,
    theta_rotation : float = 0.0
):
    
    if rir_path is not None:
        try:
            rir,fs = torchaudio.load(rir_path)
        except:
            rir,fs = sf.read(rir_path)
            rir = torch.tensor(rir).T
        assert rir.is_complex() == False , "Assumes RIR is in real SH - received complex SH"
        rir = torchaudio.functional.resample(rir,orig_freq=fs,new_freq=dry_fs)
        if hoa_t_cutoff > 0:
            print(f"Applying HOA T cutoff of {hoa_t_cutoff} ms, which corresponds to {(hoa_t_cutoff * fs)//1000} samples at {dry_fs} Hz")
            rir[4:,(hoa_t_cutoff * fs)//1000 :] = 0 # Cutoff in ms
        rir = trim_zero_columns_from_right(rir)
        
        if phi_rotation != 0.0:
            rir = rotate_sh(rir.T, azimuth_deg=phi_rotation, elevation_deg=theta_rotation).T
        
        encoded_signal = quick_convolve(s.reshape(1,-1),rir).t() # Convolve with RIR
        if encoding_type == 'complex':
            encoded_signal = real_to_complex_ambisonics(encoded_signal)
        encoded_signal = encoded_signal.T #[channels,samples]
    else:
        if dir_type == "deg":
            ph = math.radians(ph)
            th = math.radians(th)
        ph = torch.tensor(ph)
        th = torch.tensor(th)

        assert ph is not None and th is not None, "If no RIR is provided, azimuth and elevation angles must be provided"
        y = create_sh_matrix(N=sh_order, azi=ph, zen=th, sh_type=encoding_type)

        if plot:
            # The plot function mirrors the complex - not sure why, so just plot it with real (audio is fine)
            y_plot =  create_sh_matrix(N=sh_order, azi=ph, zen=th, sh_type='real')
            spa.plot.sh_coeffs(y_plot.T/y.abs().sum(), cbar=False, sh_type = 'real')

        if signal_domain == "freq":
            raise NotImplementedError
        elif signal_domain == 'time':
            s = s.to(y.dtype)
            encoded_signal =  y.reshape(-1,1) @ s.reshape(1,-1)#torch.outer(s.squeeze(),y.squeeze())
        else:
            raise ValueError(f"{signal_domain} signal dom is not known")

    return encoded_signal


def create_soundfield(
    speakers: list,
    sh_order: int = 3,
    encoding_type: str = 'complex',
    fs: int = 16000,
) -> torch.Tensor:
    """
    Build a (multi-)speaker ambisonics soundfield by encoding each speaker
    and summing them together.

    Args:
        speakers: List of speaker dicts. Each dict contains:
            - 'audio': torch.Tensor - dry mono signal ([1, T] or [T])
            - EITHER 'rir_path': str/Path - path to ambisonics RIR
              OR 'th': float, 'ph': float - DOA in radians (free-field plane wave)
            - (optional) 'gain_db': float - gain to apply in dB (default 0.0)
            - (optional) 'phi_rotation': float - azimuth rotation in degrees (default 0.0)
            - (optional) 'theta_rotation': float - elevation rotation in degrees (default 0.0)
            - (optional) 'offset_sec': float - time offset in seconds before speaker starts (default 0.0)
              The speaker's signal will be zero-padded at the beginning by offset_sec * fs samples.
        sh_order: Spherical harmonics order (default 3)
        encoding_type: 'real' or 'complex' (default 'complex')
        fs: Sample rate in Hz (default 16000)

    Returns:
        torch.Tensor of shape [C, T] - summed ambisonics signal, where C = (sh_order+1)^2
        T is the length of the longest encoded speaker signal.
    """
    encoded_signals = []
    for spk in speakers:
        audio = spk['audio'].clone()
        gain_db = spk.get('gain_db', 0.0)
        if gain_db != 0.0:
            audio = audio * 10 ** (gain_db / 20)

        enc = encode_signal(
            s=audio,
            dry_fs=fs,
            sh_order=sh_order,
            th=spk.get('th'),
            ph=spk.get('ph'),
            encoding_type=encoding_type,
            rir_path=spk.get('rir_path'),
            phi_rotation=spk.get('phi_rotation', 0.0),
            theta_rotation=spk.get('theta_rotation', 0.0),
        )  # [C, T_i]

        # Apply time offset: prepend silence so this speaker starts later
        offset_sec = spk.get('offset_sec', 0.0)
        if offset_sec > 0:
            offset_samples = int(offset_sec * fs)
            enc = torch.cat([torch.zeros(enc.shape[0], offset_samples, dtype=enc.dtype, device=enc.device), enc], dim=1)

        encoded_signals.append(enc)

    # Align lengths: pad shorter signals to the longest
    max_len = max(s.shape[1] for s in encoded_signals)
    n_ch = encoded_signals[0].shape[0]
    dtype = encoded_signals[0].dtype
    device = encoded_signals[0].device
    soundfield = torch.zeros(n_ch, max_len, dtype=dtype, device=device)
    for enc in encoded_signals:
        soundfield[:, :enc.shape[1]] += enc

    return soundfield

def encode_signal_old(
    s : torch.tensor,
    sh_order : int = 1,
    ph : Optional[float] = None,
    th : Optional[float] = None,
    type : str = "real",
    n_fft : int  = 1024,        # Number of FFT points
    plot  = False,
    rir_path : Optional[str] = None,
    hoa_t_cutoff :int = -1
):
    
    if rir_path is not None:
        try:
            rir,fs = torchaudio.load(rir_path)
        except:
            rir,fs = sf.read(rir_path)
            rir = torch.tensor(rir).T
        if hoa_t_cutoff > 0:
            rir[4:,(hoa_t_cutoff * fs)//1000 :] = 0 # Cutoff in ms
        rir = trim_zero_columns_from_right(rir)
        encoded_signal = quick_convolve(s.reshape(1,-1),rir).t() # Convolve with RIR

    else:
        assert ph is not None and th is not None, "If no RIR is provided, azimuth and elevation angles must be provided"
        y = torch.tensor(spa.sph.sh_matrix(N_sph=sh_order, azi=ph, zen=th, sh_type=type),dtype=torch.float if type == 'real' else torch.complex64)

        if plot:
            debug = torch.ones((1, 16))
            debug = debug * (4 * torch.pi) / (debug.shape[1] + 1) ** 2
            spa.plot.sh_coeffs(y, cbar=False, sh_type = type)

        if type == "complex":
            # Parameters for STFT
            hop_length = n_fft//2   # Hop length (stride)
            win_length = n_fft   # Window length
            window = torch.hann_window(win_length)  # Hann window
            s_f = torch.stft(
                s.reshape(1,-1), 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window, 
                return_complex=True,
            )
            encoded_signal = y.T.conj().unsqueeze(1) * s_f  # Shape [#Channels , #FreqBands, #TimeFrames] TODO Batches
        elif type == 'real':
            encoded_signal = torch.matmul(s.reshape(-1, 1), y.reshape(1, -1))
        else:
            raise ValueError(f"{type} SH is not known")

    return encoded_signal

def build_soundfield_from_df(speakers_df : pd.DataFrame ,ambisonic_order : int,SH_type : str = 'real',n_fft : int =  510 , num_frames : int = 256 , audio_length_sec : float = -1 , offset_audio_start_sec : float = 1):
    assert len(set(speakers_df['sr'])) == 1, "Sample rate not consistent between segments"
    anm_t = 0
    for i,row in speakers_df.iterrows():    
        audio,dry_fs = torchaudio.load(row['wav_path'])
        if dry_fs != row['sr']:
            print(f"Sample rate mismatch for {row['wav_path']}: expected {row['sr']}, got {dry_fs}. Resampling...")
            audio = torchaudio.transforms.Resample(orig_freq=dry_fs,new_freq=row['sr'])(audio)
            dry_fs = row['sr']
        samples_in_segment = int((num_frames - 1) * num_frames//2) #We assume spectogram is num_frames**2
        audio = audio.squeeze()
        if audio_length_sec > 0:
            # useful for listening test for longer audios
            signal = audio[int(row['sr'] * offset_audio_start_sec):int((audio_length_sec + offset_audio_start_sec) * row['sr']) ] 
            pad = samples_in_segment - signal.shape[0] % samples_in_segment
            if pad != 0:
                # DiffAU expects audio in sizes of samples_in_segment, so make sure we can divide it to blocks of samples_in_segment
                signal = torch.nn.functional.pad(signal,(0,pad))
        else:
            #take only a segment
            signal = audio[row['segment_num'] * samples_in_segment : min((row['segment_num'] + 1)*samples_in_segment,len(audio))]
            if len(signal) < samples_in_segment:
                signal = torch.nn.functional.pad(signal,(0,samples_in_segment -len(signal)))
        
        new_signal = encode_signal(
                    signal,
                    dry_fs= dry_fs,
                    sh_order=ambisonic_order,
                    th = math.radians(row.get('doas',[0])[0]),
                    ph = math.radians(row.get('doas',[0,0])[1]),
                    plot=False,
                    encoding_type=SH_type,
                    n_fft = n_fft,
                    rir_path = row.get('rir_path'),
        ).T
        if isinstance(anm_t,int):
            anm_t += new_signal
        else:        
            length_to_add = min(anm_t.shape[0],new_signal.shape[0])
            anm_t[:length_to_add,:] += new_signal[:length_to_add,:]
    return anm_t

def build_soundfield_from_df_v2(
    speakers_df: pd.DataFrame, 
    ambisonic_order: int, 
    SH_type: str = 'real', 
    n_fft: int = 510, 
    num_frames: int = 256, 
    audio_length_sec: float = -1, 
    offset_audio_start_sec: float = 1
):
    assert len(set(speakers_df['sr'])) == 1, "Sample rate not consistent between segments"
    sr = speakers_df['sr'].iloc[0]
    samples_in_segment = int((num_frames - 1) * num_frames // 2)
    
    # --- Pass 1: Calculate the maximum required length ---
    max_len = 0
    signal_configs = [] # Store calculated bounds to avoid re-calculating

    for i, row in speakers_df.iterrows():
        # Get metadata to find file length without loading full audio if possible
        # but for accuracy with your slicing, we check the logic:
        if audio_length_sec > 0:
            start_idx = int(sr * offset_audio_start_sec)
            end_idx = int((audio_length_sec + offset_audio_start_sec) * sr)
            current_len = end_idx - start_idx
            
            # Account for the "block-size" padding logic
            pad = samples_in_segment - (current_len % samples_in_segment)
            if pad != samples_in_segment:
                current_len += pad
        else:
            # For segment mode, length is always exactly samples_in_segment
            current_len = samples_in_segment
        
        max_len = max(max_len, current_len)
        signal_configs.append((start_idx if audio_length_sec > 0 else None))

    # --- Pass 2: Process, Pad to Max, and Sum ---
    anm_t = None
    
    for idx, (i, row) in enumerate(speakers_df.iterrows()):
        audio, _ = torchaudio.load(row['wav_path'])
        audio = audio.squeeze()
        
        if audio_length_sec > 0:
            start = int(sr * offset_audio_start_sec)
            end = int((audio_length_sec + offset_audio_start_sec) * sr)
            signal = audio[start:end]
        else:
            start = row['segment_num'] * samples_in_segment
            end = min((row['segment_num'] + 1) * samples_in_segment, len(audio))
            signal = audio[start:end]
            
        # Global Padding: Pad every signal to the same global maximum
        # This ensures encode_signal returns the same shape for everyone
        padding_needed = max_len - signal.shape[0]
        if padding_needed > 0:
            signal = torch.nn.functional.pad(signal, (0, padding_needed))
            
        encoded = encode_signal(
            signal,
            ambisonic_order,
            th = math.radians(row.get('doas',[0])[0]),
            ph = math.radians(row.get('doas',[0,0])[1]),
            plot=False,
            encoding_type=SH_type,
            n_fft=n_fft,
            rir_path=row.get('rir_path')
        ).T

        if anm_t is None:
            anm_t = torch.zeros_like(encoded)
        
        anm_t += encoded
        
    return anm_t

def MagLS(hrtf_time : torch.tensor,fs : int,Yp : torch.tensor, N_fft : int = 2048, freq_cutoff : int = 2000,test = None,return_freq : bool = False,**kwargs):
#   Implementation based on Ambisonics book (https://link.springer.com/book/10.1007/978-3-030-17207-7)
#   Eqs.(4.57)-(4.59)

    fc_low = freq_cutoff * 2 **(-1/2)
    fc_high = freq_cutoff * 2 **(1/2)
    f_bin = fs/N_fft
    low_fc_index = int(fc_low/f_bin)
    high_fc_index = int(fc_high/f_bin)
    hrtf_time_padded = F.pad(hrtf_time, (0, 0, 0, N_fft - hrtf_time.shape[1]))
    HRTF_f= torch.fft.rfft(hrtf_time_padded, n=N_fft, dim=1)
    HRTF_f_l = HRTF_f[:, :, 0]
    HRTF_f_r = HRTF_f[:, :, 1]

    Yp_inv = torch.linalg.pinv(Yp)

    num_freq_bins = N_fft//2  + 1
    num_channels = Yp.shape[1]
    H_l_nm_f_MagLS = torch.zeros(num_channels,num_freq_bins,dtype=HRTF_f.dtype)
    H_r_nm_f_MagLS = torch.zeros(num_channels,num_freq_bins,dtype=HRTF_f.dtype)


    # When < fc_low do regular LS
    H_l_nm_f_MagLS[:,:low_fc_index] = Yp_inv @ HRTF_f_l[:,:low_fc_index]
    H_r_nm_f_MagLS[:,:low_fc_index] = Yp_inv @ HRTF_f_r[:,:low_fc_index]

    for freq_id in range(low_fc_index,high_fc_index + 1):
        alpha = 0.5 + math.log2(freq_id * f_bin / freq_cutoff)

        phi_left = torch.angle(Yp @ H_l_nm_f_MagLS[:,freq_id-1])
        H_l_nm_f_MagLS[:,freq_id] = alpha * Yp_inv @ (torch.abs(HRTF_f_l[:,freq_id]).to(torch.cdouble) * torch.exp(1j*phi_left)) + (1-alpha) * Yp_inv @ HRTF_f_l[:,freq_id] # Eq. (4.59)

        phi_right = torch.angle(Yp @ H_r_nm_f_MagLS[:,freq_id-1])
        H_r_nm_f_MagLS[:,freq_id] = alpha * Yp_inv @ (torch.abs(HRTF_f_r[:,freq_id]).to(torch.cdouble) * torch.exp(1j*phi_right)) + (1-alpha) * Yp_inv @ HRTF_f_r[:,freq_id] # Eq. (4.59)

    for freq_id in range(high_fc_index,HRTF_f.shape[1]):
        phi_left = torch.angle(Yp @ H_l_nm_f_MagLS[:,freq_id-1])
        H_l_nm_f_MagLS[:,freq_id] = Yp_inv @ (torch.abs(HRTF_f_l[:,freq_id]).to(torch.cdouble) * torch.exp(1j*phi_left))

        phi_right = torch.angle(Yp @ H_r_nm_f_MagLS[:,freq_id-1])
        H_r_nm_f_MagLS[:,freq_id] = Yp_inv @ (torch.abs(HRTF_f_r[:,freq_id]).to(torch.cdouble) * torch.exp(1j*phi_right))


    H_l_nm_f_MagLS = sh_freq_complement(H_l_nm_f_MagLS)
    H_r_nm_f_MagLS = sh_freq_complement(H_r_nm_f_MagLS)
    if return_freq:
        return H_l_nm_f_MagLS, H_r_nm_f_MagLS
    else:
        H_l_nm_t_MagLS = torch.roll(torch.fft.ifft(H_l_nm_f_MagLS, n=N_fft, dim=1),shifts = 250)
        H_r_nm_t_MagLS = torch.roll(torch.fft.ifft(H_r_nm_f_MagLS, n=N_fft, dim=1),shifts = 250)
        return H_l_nm_t_MagLS, H_r_nm_t_MagLS

def render_binaural(
    signal: torch.tensor,
    signal_fs: int,
    theta: Optional[float] = None,
    phi: Optional[float] = None,
    sh_order: Optional[int] = 3,
    HRTF_time: Optional[torch.tensor] = None,
    hrtf_fs: Optional[int] = None,
    play : bool = True,
    MagLS_flag : bool = False,
    save_path : Optional[str] = None,
    name_suffix : Optional[str] = None,
    nfft = 2048,
    upsample_fs = 48000,
    **kwargs
):
    if signal_fs < upsample_fs:
        print(f"Upsampling signal from {signal_fs} to {upsample_fs}")
        is_complex = False
        if torch.is_complex(signal):
            signal = complex_to_real_ambisonics(signal)
            is_complex = True
        resampler = torchaudio.transforms.Resample(orig_freq=signal_fs, new_freq=upsample_fs)
        signal = resampler(signal)
        if is_complex:
            signal = real_to_complex_ambisonics(signal)
        signal_fs = upsample_fs
    # If no HRTF is provided, load the default HRTF
    if HRTF_time is None:
        if hrtf_fs is None:
            if signal_fs == 16000:
                _hrtf = loadmat(r"C:\Users\amitmils\Documents\Repo\DiffAU\HRTF_time_16k.mat")
            elif signal_fs == 48000:
                _hrtf = loadmat(Path().cwd().parent / "HRTF_time_48k.mat")
            else:
                raise ValueError(f"Unsupported signal fs {signal_fs}. Supported fs are 16kHz and 48kHz.")
            hrtf_fs = signal_fs
        else:
            if hrtf_fs == 16000:
                _hrtf = loadmat(Path().cwd().parent / "HRTF_time_16k.mat")
            elif hrtf_fs == 48000:
                _hrtf = loadmat(Path().cwd().parent / "HRTF_time_48k.mat")
            else:
                raise ValueError(f"Unsupported HRTF fs {hrtf_fs}. Supported fs are 16kHz and 48kHz.")
        HRTF_time = _hrtf['HRTF_time']

    elif HRTF_time.shape[0] != 2702:
        raise NotImplementedError("HRTF not in Lebedev grid")

    # If dry signal, encode to ambisonic
    if signal.dim() == 1 or (signal.dim() == 2 and signal.shape[0] == 1):
        raise DeprecationWarning("Use encode_signal - this might not work as expected") #TODO call encode_signal 
        # Dry signal
        if hrtf_fs != signal_fs:
            print(f"Resampling signal from {signal_fs} to {hrtf_fs}")
            resampler = torchaudio.transforms.Resample(orig_freq=signal_fs, new_freq=hrtf_fs)
            signal = resampler(signal)
        signal = signal.squeeze()
        sh_vec = create_sh_matrix(sh_order,zen=torch.tensor(math.radians(theta)),azi=torch.tensor(math.radians(phi)),sh_type = 'real').reshape(-1)  # Shape: [num_channels]
        ambi_signal = torch.outer(sh_vec,signal)
        print(f"Encoded signal to order {sh_order} SH")
    elif signal.dim() == 2:
        sh_order = int(math.sqrt(signal.shape[0]) - 1)
        ambi_signal = signal
        assert signal_fs == hrtf_fs, f"Signal fs {signal_fs} does not match HRTF fs {hrtf_fs}"
        print(f"Signal is already in order {sh_order} SH - ignoring inputs theta, phi, sh_order")
    else:
        raise ValueError(f"Signal dimension {signal.dim()} not supported. Expected 1D or 2D tensor.")
    
    P_th, P_ph, _ = grids.create_grid('lebedev')
    # The signal is assumed to be in the TIME domain. If its complex --> Complex SH otherwise its Real SH
    sh_type = 'complex' if torch.is_complex(ambi_signal) else 'real'
    print(f"Encoding with {sh_type} SH")
    if MagLS_flag == True and sh_type == 'real':
        raise TypeError("For MagLS signal must be encoded with complex SH")#TODO convert real SH to Complex
    Yp = create_sh_matrix(sh_order,zen=P_th,azi=P_ph,sh_type = sh_type)

    ambi_signal_f = torch.stft(
                    ambi_signal, 
                    n_fft=nfft, 
                    hop_length=nfft//2, 
                    win_length=nfft, 
                    window=torch.hann_window(nfft), 
                    return_complex=True,
                    onesided=False if sh_type == 'complex' else True
                )
    if MagLS_flag:
        test_path = None
        if kwargs.get('old_version',False):
            SH_HRTF_left, SH_HRTF_right =  MagLS(torch.tensor(HRTF_time),fs=hrtf_fs,Yp=Yp,test=test_path,Y_p_real=create_sh_matrix(sh_order,zen=P_th,azi=P_ph,sh_type="real"),return_freq=False,**kwargs)
            SH_HRTF_left = complex_to_real_ambisonics(SH_HRTF_left.T).T
            SH_HRTF_right = complex_to_real_ambisonics(SH_HRTF_right.T).T
        else:
            SH_HRTF_left_f, SH_HRTF_right_f =  MagLS(torch.tensor(HRTF_time),fs=hrtf_fs,Yp=Yp,test=test_path,Y_p_real=create_sh_matrix(sh_order,zen=P_th,azi=P_ph,sh_type="real"),return_freq=True,**kwargs)

    else:
        Yp_inv = torch.linalg.pinv(Yp)
        SH_HRTF_left = Yp_inv @ HRTF_time[:, :, 0]  # shape: [num_channels, num_samples]
        SH_HRTF_right = Yp_inv @ HRTF_time[:, :, 1]  # shape: [num_channels, num_samples]
        if torch.is_complex(ambi_signal) or torch.is_complex(Yp):
            fft = torch.fft.fft
        else:
            fft = torch.fft.rfft
        SH_HRTF_left_f = fft(SH_HRTF_left, n=nfft,dim=1)
        SH_HRTF_right_f = fft(SH_HRTF_right, n=nfft,dim=1)

    if kwargs.get('old_version',False):
        print("Using old version of convolution")
        y_left = torch.sum(quick_convolve(ambi_signal,SH_HRTF_left),dim=0)
        y_right = torch.sum(quick_convolve(ambi_signal,SH_HRTF_right),dim=0)
    else:
        y_left =  torch.istft((ambi_signal_f * SH_HRTF_left_f[...,None]).sum(0), n_fft=nfft, 
                hop_length=nfft//2, 
                win_length=nfft, 
                window=torch.hann_window(nfft))
        y_right =  torch.istft((ambi_signal_f * SH_HRTF_right_f[...,None]).sum(0), n_fft=nfft, 
                hop_length=nfft//2, 
                win_length=nfft, 
                window=torch.hann_window(nfft))


    # --- Stack and normalize ---
    binaural = torch.stack([y_left, y_right])      # [2, L]
    binaural /=binaural.abs().max()

    if play:
        sd.play(binaural.T, samplerate=hrtf_fs)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        name = f"binaural_{sh_order}SH_{'MagLS' if MagLS_flag else 'LS'}"
        if name_suffix is not None:
            name = f"{name}_{name_suffix}"
        torchaudio.save(os.path.join(save_path,f"{name}.wav"), binaural, hrtf_fs)
        print(f"Saved binaural audio to {os.path.join(save_path,f'{name}.wav')}")

    return binaural

def build_soundfield_from_dataset_df_row(row, speaker_data_base_path: str, rir_data_base_path: str, 
                                         hoa_t_cutoff: int = -1, random_gain_augmentation: bool = False):
    """
    Build soundfield from dataset dataframe row.
    
    Args:
        row: Dataframe row containing speaker and RIR information
        speaker_data_base_path: Base path to speaker audio files
        rir_data_base_path: Base path to RIR files
        hoa_t_cutoff: Cutoff time in ms for HOA components (-1 for no cutoff)
        random_gain_augmentation: If True, use gain values from metadata (row['gain_db'])
    
    Returns:
        signal tensor (gain values are already stored in metadata, no need to return them)
    """
    anm_t_signals = list()
    
    # Get gain values from metadata (will be [0.0, 0.0, ...] if augmentation is disabled)
    gain_db_list = row.get('gain_db', [0.0] * row.num_speakers)
    
    # Get phi rotation value from metadata (single value for entire soundfield, default 0.0)
    # This is now a single float value, not a list - all sources share the same rotation
    phi_rotation = row.get('phi_rotation', 0.0)
    
    for idx, (speaker_path, rir_path, (start, end)) in enumerate(zip(row['speaker_rel_path'], row['rir_rel_path'], row['timestamp_sec'])):
        full_speaker_path = os.path.join(speaker_data_base_path, speaker_path)
        full_rir_path = os.path.join(rir_data_base_path, rir_path)
        if not os.path.exists(full_speaker_path):
            raise FileNotFoundError(f"Speaker file {full_speaker_path} does not exist")
        if not os.path.exists(full_rir_path):
            raise FileNotFoundError(f"RIR file {full_rir_path} does not exist")
        
        dry_signal, fs = torchaudio.load(full_speaker_path)
        segment = dry_signal[:, round(start*fs):round(end*fs)]
        
        # Apply pre-determined gain from metadata
        gain_db = gain_db_list[idx]
        if gain_db != 0.0:
            gain_linear = 10 ** (gain_db / 20)
            segment = segment * gain_linear
        
        # Apply the same phi rotation to all sources in the soundfield
        anm_t_signals.append(encode_signal(s=segment, dry_fs=fs, rir_path=full_rir_path, hoa_t_cutoff=hoa_t_cutoff, phi_rotation=phi_rotation).T)

    signal = torch.stack(anm_t_signals, dim=0).sum(0)
    
    return signal

def channelwise_convolve(A, B):
    """
    Convolve each channel of A with the corresponding channel of B.
    Args:
        A: Tensor (C, T_a)
        B: Tensor (C, T_b)
    Returns:
        Tensor (C, T_out)
    """
    C, T_a = A.shape
    _, T_b = B.shape
    T_out = T_a + T_b - 1

    A_ = A.unsqueeze(0)  # (1, C, T_a)
    B_ = B.flip(dims=[1]).unsqueeze(1)  # (C, 1, T_b) flipped for conv
    
    out = F.conv1d(A_, B_, groups=C) 
    start = (T_b - 1) // 2
    end = start + T_a # (1, C, T_out)
    return out.squeeze(0)[..., start:end]
