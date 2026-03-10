import os
import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy import special as scyspecial
import math

def spherical_to_cartesian(r, theta, phi):
    """Converts spherical coordinates (r, theta, phi) to cartesian coordinates (x, y, z)"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def mean_conf_int(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

class Method():
    def __init__(self, name, base_dir, metrics):
        self.name = name
        self.base_dir = base_dir
        self.metrics = {} 
        
        for i in range(len(metrics)):
            metric = metrics[i]
            value = []
            self.metrics[metric] = value 
            
    def append(self, matric, value):
        self.metrics[matric].append(value)

    def get_mean_ci(self, metric):
        return mean_conf_int(np.array(self.metrics[metric]))

def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

def sdr_batch(s, s_hat):
    """
    Compute regular SDR (not scale-invariant) for batched signals using PyTorch.
    
    Args:
        s: torch.Tensor of shape (B, T) — ground truth signals
        s_hat: torch.Tensor of shape (B, T) — estimated signals
    
    Returns:
        sdr_values: torch.Tensor of shape (B,) — SDR per signal in the batch
    """
    eps = 1e-8
    s = s.float()
    s_hat = s_hat.float()

    noise = s - s_hat
    signal_power = torch.sum(s ** 2, dim=1)              # (B,)
    noise_power = torch.sum(noise ** 2, dim=1)           # (B,)
    
    sdr = 10 * torch.log10((signal_power + eps) / (noise_power + eps))
    return sdr

def si_sdr_batch(s, s_hat):
    """
    Compute SI-SDR for batched signals using PyTorch.
    
    Args:
        s: torch.Tensor of shape (B, T) — ground truth signals
        s_hat: torch.Tensor of shape (B, T) — estimated signals
    
    Returns:
        si_sdr_values: torch.Tensor of shape (B,) — SI-SDR per signal in the batch
    """
    eps = 1e-8
    s = s.float()
    s_hat = s_hat.float()

    dot = torch.sum(s * s_hat, dim=1)                    # (B,)
    s_energy = torch.sum(s ** 2, dim=1)                  # (B,)
    alpha = dot / (s_energy + eps)                       # (B,)
    
    target = alpha.unsqueeze(1) * s                      # (B, T)
    noise = target - s_hat                               # (B, T)
    
    target_energy = torch.sum(target ** 2, dim=1)        # (B,)
    noise_energy = torch.sum(noise ** 2, dim=1)          # (B,)
    
    si_sdr = 10 * torch.log10((target_energy + eps) / (noise_energy + eps))
    return si_sdr


import torch

def stft_sdr_per_band(s, s_hat, eps=1e-8):
    """
    Computes STFT SDR per frequency band.

    Args:
        s:     torch complex tensor [B, C, T, F] or [C, T, F]
        s_hat: torch complex tensor [B, C, T, F] or [C, T, F]
        eps:   numerical stability term

    Returns:
        sdr_per_band: torch tensor [B, F] — SDR per frequency bin
    """
    # Handle batch dimension
    if s.dim() == 3:
        s = s.unsqueeze(0)     # [1, C, T, F]
        s_hat = s_hat.unsqueeze(0)
        had_no_batch = True
    else:
        had_no_batch = False

    # 1. Calculate the squared magnitude (power) for signal and error
    # We keep the shape [B, C, T, F]
    signal_power_map = torch.abs(s) ** 2
    error_power_map = torch.abs(s_hat - s) ** 2

    # 2. Sum over Channels (dim 1) and Time (dim 23)
    # This leaves us with [B, F]
    signal_band_power = torch.sum(signal_power_map, dim=(1, 3))
    noise_band_power = torch.sum(error_power_map, dim=(1, 3)) + eps

    # 3. Calculate SDR per band
    sdr_per_band = 10 * torch.log10(signal_band_power / noise_band_power)

    if had_no_batch:
        sdr_per_band = sdr_per_band.squeeze(0)

    return sdr_per_band

def stft_sdr(s, s_hat, eps=1e-8,full_audio_sdr=False):
    """
    Computes STFT SNR for complex-valued inputs.

    Args:
        s:     torch complex tensor [B, C, T, F] — reference
        s_hat: torch complex tensor [B, C, T, F] — estimated
        eps:   numerical stability term

    Returns:
        snr: torch tensor [B, C] — SNR per channel per batch
    """
    if s.dim() == 3:
        no_batch_dim = True
        s = torch.unsqueeze(s, dim=0)
        s_hat = torch.unsqueeze(s_hat, dim=0)
    else:
        no_batch_dim = False

    B, C, T, F = s.shape
    s_flat = s.contiguous().view(B, C, -1)         # [B, C, T*F]
    s_hat_flat = s_hat.contiguous().view(B, C, -1) # [B, C, T*F]

    error = s_hat_flat - s_flat                    # [B, C, T*F]

    signal_power_per_channel = torch.sum(torch.abs(s_flat) ** 2, dim=-1)     # [B, C]
    noise_power_per_channel = torch.sum(torch.abs(error) ** 2, dim=-1) + eps # [B, C]

    signal_power = torch.sum(signal_power_per_channel,dim=-1)   # [B]
    noise_power = torch.sum(noise_power_per_channel,dim=-1)     # [B]
    sdr_total = 10 * torch.log10(signal_power / noise_power)    # [B]


    sdr_per_channel = 10 * torch.log10(signal_power_per_channel / noise_power_per_channel)           # [B]


    if no_batch_dim:
        sdr_total = torch.squeeze(sdr_total, dim=0)
        sdr_per_channel = torch.squeeze(sdr_per_channel, dim=0)

    if full_audio_sdr:
        return sdr_per_channel,sdr_total
    else:
        return sdr_per_channel

def stft_si_sdr(s, s_hat, eps=1e-8,full_audio_sdr=False):
    """
    Computes STFT SI-SDR for complex-valued inputs.
    
    Args:
        s:     torch complex tensor [B, C, T, F] — reference
        s_hat: torch complex tensor [B, C, T, F] — estimated
        eps:   numerical stability term

    Returns:
        si_sdr: torch tensor [B, C] — SI-SDR per channel per batch
    """
    if s.dim() == 3:
        no_batch_dim = True
        s = torch.unsqueeze(s,dim = 0)
    else:
        no_batch_dim = False

    B, C, T, F = s.shape
    s_flat = s.contiguous().view(B, C, -1)             # [B, C, T*F]
    s_hat_flat = s_hat.contiguous().view(B, C, -1)     # [B, C, T*F]

    dot = torch.sum(s_hat_flat.conj() * s_flat, dim=-1)          # [B, C]
    denom = torch.sum(s_flat.conj() * s_flat, dim=-1) + eps      # [B, C]
    alpha = dot / denom                                          # [B, C]

    projection = alpha.unsqueeze(-1) * s_flat                    # [B, C, T*F]
    error = projection - s_hat_flat

    signal_power_per_channel = torch.sum(torch.abs(projection) ** 2, dim=-1) # [B, C]
    noise_power_per_channel = torch.sum(torch.abs(error) ** 2, dim=-1) + eps

    signal_power = torch.sum(signal_power_per_channel,dim=-1)   # [B]
    noise_power = torch.sum(noise_power_per_channel,dim=-1)     # [B]
    si_sdr_total = 10 * torch.log10(signal_power / noise_power)    # [B]

    si_sdr_per_channel = 10 * torch.log10(signal_power_per_channel / noise_power_per_channel)        # [B, C]

    if no_batch_dim:
        si_sdr_total = torch.squeeze(si_sdr_total, dim=0)
        si_sdr_per_channel = torch.squeeze(si_sdr_per_channel, dim=0)

    if full_audio_sdr:
        return si_sdr_per_channel,si_sdr_total
    else:
        return si_sdr_per_channel

def snr_dB(s,n):
    s_power = 1/len(s)*np.sum(s**2)
    n_power = 1/len(n)*np.sum(n**2)
    snr_dB = 10*np.log10(s_power/n_power)
    return snr_dB

def pad_spec(Y, mode="zero_pad"):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    if mode == "zero_pad":
        pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    elif mode == "reflection":
        pad2d = torch.nn.ReflectionPad2d((0, num_pad, 0,0))
    elif mode == "replication":
        pad2d = torch.nn.ReplicationPad2d((0, num_pad, 0,0))
    else:
        raise NotImplementedError("This function hasn't been implemented yet.")
    return pad2d(Y)

def set_torch_cuda_arch_list():
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs found.")
        return
    
    num_gpus = torch.cuda.device_count()
    compute_capabilities = []

    for i in range(num_gpus):
        cc_major, cc_minor = torch.cuda.get_device_capability(i)
        cc = f"{cc_major}.{cc_minor}"
        compute_capabilities.append(cc)
    
    cc_string = ";".join(compute_capabilities)
    os.environ['TORCH_CUDA_ARCH_LIST'] = cc_string
    print(f"Set TORCH_CUDA_ARCH_LIST to: {cc_string}")

def create_sh_matrix(N, azi, zen, sh_type="real"):
    """
    Create a spherical harmonics matrix.

    Parameters:
    N (int): The order of the spherical harmonics.
    azi (torch.tensor): Array of azimuthal angles in radians.
    zen (torch.tensor): Array of zenith angles in radians.
    type (str, optional): Type of spherical harmonics ('complex' or 'real'). Default is 'complex'.

    Returns:
    torch.tensor: The spherical harmonics matrix.
    """
    azi = azi.reshape(-1)
    zen =zen.reshape(-1)

    if azi.ndim == 0:
        Q = 1
    else:
        Q = len(azi)
    if sh_type == 'complex':
        Ymn = torch.zeros([Q, (N+1)**2], dtype=torch.complex128)
    elif sh_type == 'real':
        Ymn = torch.zeros([Q, (N+1)**2], dtype=torch.float64)
    else:
        raise ValueError('sh_type unknown.')

    idx = 0
    for n in range(N+1):
        for m in range(-n, n+1):
            Ymn_complex = scyspecial.sph_harm(m, n, azi, zen)
            if sh_type == 'complex':
                Ymn[:, idx] = Ymn_complex
            elif sh_type == 'real':
                if m == 0:
                    Ymn[:, idx] = Ymn_complex.real
                if m < 0:
                    Ymn[:, idx] = np.sqrt(2) * Ymn_complex.imag * (-1) ** abs(m) 
                if m > 0:
                    Ymn[:, idx] = np.sqrt(2)  * Ymn_complex.real * (-1) ** m
            idx += 1
    return Ymn


def plot_ambi_energy_2d(rir : torch.tensor, step_size=10, three_D = False,two_D = True):
    """Visualizes RIR energy distribution over a 3D sphere and 2D map.
        Assumes signal is in time domain and ACN format."""
    
    if rir.is_complex():
        rir = complex_to_real_ambisonics(rir.T).T
    rir = torch.tensor(rir,dtype=torch.float64)
    order = int(np.sqrt(rir.shape[0]) - 1)  # Assuming rir is in ACN format

    azi = torch.linspace(-np.pi, np.pi, step_size, dtype=torch.float32)
    colat = torch.linspace(0, np.pi, step_size, dtype=torch.float32)
    C, A = torch.meshgrid(colat,azi, indexing='ij')
    

    Y_grid = create_sh_matrix(order, A.flatten(), C.flatten()).to(rir.dtype)

    decoded = Y_grid @ rir

    energy_map = torch.sum(decoded**2, dim=1).view(step_size, step_size)
    energy_map = energy_map / (torch.max(energy_map) + 1e-9)

    # Set up the 3D plot
    if three_D:
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(121, projection='3d')

        # Create a spherical surface
        theta, phi = np.mgrid[0:np.pi:step_size*1j, 0:2*np.pi:step_size*1j]
        x, y, z = spherical_to_cartesian(1, theta, phi)

        # Plot the energy map on the 3D surface - aligning it to ITU format (90 is left and -90 is right)
        ax.plot_surface(y, -x, z, facecolors=plt.cm.viridis(energy_map), rstride=1, cstride=1, alpha=0.7, antialiased=True)
        ax.set_title("Energy Distribution (3D Sphere)")
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_zlabel("Z")


    if two_D:
        fig = plt.figure(figsize=(12, 8))

        # Convert radians to degrees for 2D plot
        azi_deg = np.degrees(azi)
        colat_deg = np.degrees(colat)
        
        # 2D Plot with degrees
        ax2 = fig.add_subplot(122)
        c = ax2.pcolormesh(azi_deg, colat_deg, energy_map, shading='auto', cmap='viridis')
        ax2.set_title(f"Energy Distribution  - Order : {order}")
        ax2.set_xlabel("Azimuth (degrees)")
        ax2.set_ylabel("Colatitude (degrees)")
        fig.colorbar(c, ax=ax2)

    plt.show()
    return energy_map

def quick_convolve(x,h,remove_delay_in_filter = True):
    """
    Perform convolution of two tensors using FFT for efficiency.
    Parameters:
    x (torch.Tensor): Input tensor of shape (C, T)   C is channel dimension and T is time dimension.
    h (torch.Tensor): Filter tensor of shape (C, K) where  C is channel dimension  and K is filter length.    
    """
    #TODO turn to overlap+app for long signals
    #TODO support bathch dimension

    if x.is_complex():
        fft = torch.fft.fft
        ifft = torch.fft.ifft
    else:
        fft = torch.fft.rfft
        ifft = torch.fft.irfft

    added_batch_dim = False
    if x.dim() == 2:
        x = x.unsqueeze(0)  # Add batch dimension
        added_batch_dim = True
    if h.dim() == 2:
        h = h.unsqueeze(0)  # Add batch dimension

    B, C, T = x.shape
    filter_batch, _, K = h.shape
    assert filter_batch == 1, "Filter batch size must be 1."
    L_fft = 2**((T + K - 1) - 1).bit_length()  # Next power of 2 for FFT efficiency

    if remove_delay_in_filter:
        #currently only supports B=1
        eps = 1e-8 
        nonzero_mask = (h.abs() > eps)
        first_nonzero = torch.argmax(nonzero_mask.float(), dim=-1)
        min_start = first_nonzero.min(dim=1).values
        h = h[..., min_start.item():]
    h = h.expand(B, -1, -1)

    X = fft(x, n=L_fft)
    H = fft(h, n=L_fft)      
    Y = X * H    
    y = ifft(Y, n=L_fft)[...,:T] # Note to self, this might mean we kill some reverb tail - but if we assume that the last offset is <<T then its ok

    if added_batch_dim:
        y = y.squeeze(0)
    return y

def acn_index(n, m):
    return n**2 + n + m


def sh_freq_complement(pos_freq : torch.tensor):
    """
    Restores negatives frequencies of a signal thats freq SH domain
    pos_freq : torch.tensor - [channels,...]
    """

    N = int(np.sqrt(pos_freq.shape[0]) - 1)
    pos_freq_conj_flipped = pos_freq[:, 1:-1].flip(-1).conj()
    neg_freq = torch.empty_like(pos_freq_conj_flipped)
    for n in range(N+1):
        for m in range(-n, n+1):
            idx = acn_index(n, m)
            idx_neg = acn_index(n, -m)
            neg_freq[idx,:] = (-1)**m * pos_freq_conj_flipped[idx_neg,:]
    full_freq = torch.cat([pos_freq, neg_freq], dim=1)
    return full_freq


def complex_to_real_ambisonics(a_complex: torch.Tensor) -> torch.Tensor:
    """
    Transforms complex SH to real SH - for real signals only!
    a_complex : torch.Tensor - [samples,channels]
    """
   
    #basic sanity tests... 
    transposed = False
    if  a_complex.shape[1] > a_complex.shape[0]:
        transposed = True
        a_complex = a_complex.T
    assert math.sqrt(a_complex.shape[1]) == int(math.sqrt(a_complex.shape[1])), f"SH order is not an integer, received {a_complex.shape}- expects [amples,channels]"
    N =  int(math.sqrt(a_complex.shape[1])) - 1
    # create output tensor, real dtype
    a_real = torch.zeros_like(a_complex.real)

    for n in range(N+1):
        for m in range(-n, n+1):
            idx = acn_index(n, m)
            if m > 0:
                a_real[:,idx] = math.sqrt(2)  * a_complex[:,idx].real * (-1) ** m
            elif m == 0:
                a_real[:,idx] = a_complex[:,idx].real
            else:  # m < 0
                idx_pos = acn_index(n, m) #IS this inccorect? should it be -m?
                a_real[:,idx] = math.sqrt(2) * a_complex[:,idx_pos].imag * (-1) ** m

    if transposed:
        a_real = a_real.T
    return a_real


def real_to_complex_ambisonics(a_real: torch.Tensor) -> torch.Tensor:
    """
    Transforms real SH to complex SH
    a_real : torch.Tensor - [samples,channels]
    """
    #basic sanity tests... 
    transposed = False  
    if  a_real.shape[1] > a_real.shape[0]:
        transposed = True
        a_real = a_real.T
    assert math.sqrt(a_real.shape[1]) == int(math.sqrt(a_real.shape[1])), f"SH order is not an integer, received {a_real.shape}- expects [num_samples,channels]"
    N =  int(math.sqrt(a_real.shape[1]) - 1)

    # raise NotImplementedError("This function is not working properly.")
    # create output tensor, complex dtype
    a_complex = torch.zeros(*a_real.shape, dtype=torch.cfloat, device=a_real.device)

    for n in range(N+1):
        for m in range(-n, n+1):
            idx = acn_index(n, m)
            if m > 0:
                idx_pos = acn_index(n, m)
                idx_neg = acn_index(n, -m)
                real_part = a_real[:,idx_pos]
                imag_part = (-1) ** m * a_real[:,idx_neg]
                normalizing = (-1) ** m / math.sqrt(2)
                a_complex[:,idx_pos] = (real_part - 1j * imag_part) * normalizing
                a_complex[:,idx_neg] = ((-1)**m) * torch.conj(a_complex[:,idx_pos])
            elif m == 0:
                a_complex[:,idx] = a_real[:,idx] + 0j

    if transposed:
        a_complex = a_complex.T
    return a_complex

def wigner_d_matrix(n, beta):
    """
    Compute the Wigner d-matrix for order n and rotation angle beta (elevation).
    
    Parameters:
    n (int): The SH order.
    beta (float): Rotation angle around Y-axis in radians.
    
    Returns:
    torch.Tensor: Wigner d-matrix of shape [(2n+1), (2n+1)].
    """
    # Initialize the d-matrix
    d_matrix = torch.zeros((2*n + 1, 2*n + 1), dtype=torch.float64)
    
    # Compute using the formula with associated Legendre polynomials
    for m in range(-n, n+1):
        for mp in range(-n, n+1):
            # Map indices from [-n, n] to [0, 2n]
            i = m + n
            j = mp + n
            
            # Use the symmetry and recurrence relations
            # For simplicity, we use scipy's rotation for accurate computation
            from scipy.spatial.transform import Rotation as R
            
            # Create a rotation around Y-axis
            rot = R.from_euler('y', beta)
            
            # Compute Wigner D-matrix element using spherical harmonics rotation
            # D^n_{m,m'}(α,β,γ) = exp(-i*m*α) * d^n_{m,m'}(β) * exp(-i*m'*γ)
            # For d-matrix (β only), we evaluate at specific angles
            
            # Alternative: Use explicit formula
            cos_half = np.cos(beta / 2)
            sin_half = np.sin(beta / 2)
            
            # Compute the d-matrix element using the formula
            k_min = max(0, mp - m)
            k_max = min(n + mp, n - m)
            
            d_val = 0.0
            for k in range(k_min, k_max + 1):
                coeff = ((-1) ** (m - mp + k) * 
                        np.sqrt(math.factorial(n + m) * math.factorial(n - m) * 
                               math.factorial(n + mp) * math.factorial(n - mp)))
                denom = (math.factorial(n + mp - k) * math.factorial(n - m - k) * 
                        math.factorial(k) * math.factorial(k + m - mp))
                
                d_val += (coeff / denom * 
                         cos_half ** (2*n + mp - m - 2*k) * 
                         sin_half ** (m - mp + 2*k))
            
            d_matrix[i, j] = d_val
    
    return d_matrix

def build_sh_rotation_matrix(N, azimuth_deg, elevation_deg):
    """
    Build the full SH rotation matrix for order N using Wigner D-matrices.
    
    Parameters:
    N (int): Maximum SH order.
    azimuth_deg (float): Rotation angle around Z-axis in degrees.
    elevation_deg (float): Rotation angle around Y-axis in degrees.
    
    Returns:
    torch.Tensor: Rotation matrix of shape [(N+1)^2, (N+1)^2] in ACN format.
    """
    # Convert degrees to radians
    alpha = np.deg2rad(-azimuth_deg)  # azimuth (Z-axis rotation) 90 degress to the right
    beta = np.deg2rad(elevation_deg)  # elevation (Y-axis rotation)
    gamma = 0.0  # third Euler angle (usually 0 for azimuth/elevation)
    
    n_channels = (N + 1) ** 2
    R_matrix = torch.zeros((n_channels, n_channels), dtype=torch.cfloat)
    
    # Build rotation matrix for each order
    for n in range(N + 1):
        # Get Wigner d-matrix for this order
        d_n = wigner_d_matrix(n, beta)
        
        # Apply phase factors for full Wigner D-matrix
        # D^n_{m,m'}(α,β,γ) = exp(-i*m*α) * d^n_{m,m'}(β) * exp(-i*m'*γ)
        for m in range(-n, n + 1):
            for mp in range(-n, n + 1):
                # ACN indices
                idx_m = acn_index(n, m)
                idx_mp = acn_index(n, mp)
                
                # Wigner d-matrix indices
                i = m + n
                j = mp + n
                
                # Full Wigner D-matrix with phase factors
                phase = np.exp(-1j * m * alpha) * np.exp(-1j * mp * gamma)
                R_matrix[idx_m, idx_mp] = phase * d_n[i, j]
    
    return R_matrix

def rotate_sh(sh_signal, azimuth_deg = 0, elevation_deg = 0, sh_order=None):
    """
    Rotate a spherical harmonics signal in the SH domain.
    
    Parameters:
    sh_signal (torch.Tensor): Input signal of shape [n_samples, n_channels] in real SH (ACN format).
    azimuth_deg (float): Rotation angle around Z-axis in degrees (phi).
                         Example: 90 degrees rotates 90° in azimuth.
    elevation_deg (float): Rotation angle around Y-axis in degrees (theta).
    sh_order (int, optional): SH order. Auto-detected from channel count if not provided.
    
    Returns:
    torch.Tensor: Rotated SH signal of shape [n_samples, n_channels] in real SH (ACN format).
    
    Example:
        # Rotate by 90 degrees in azimuth
        rotated = rotate_sh(sh_signal, azimuth_deg=90, elevation_deg=0)
    """
    # Validate input shape
    assert sh_signal.dim() == 2, "Input must be 2D: [n_samples, n_channels]"
    assert sh_signal.shape[1] < sh_signal.shape[0], "Channels should be in second dimension"
    
    n_samples, n_channels = sh_signal.shape
    
    # Determine SH order
    if sh_order is None:
        sh_order = int(np.sqrt(n_channels) - 1)
        assert (sh_order + 1) ** 2 == n_channels, f"Invalid channel count {n_channels} for SH signal"
    
    # Convert to complex SH (Wigner D-matrices work in complex domain)
    sh_complex = real_to_complex_ambisonics(sh_signal)
    
    # Build rotation matrix
    R_matrix = build_sh_rotation_matrix(sh_order, azimuth_deg, elevation_deg)
    
    # Move to same device and dtype as input
    R_matrix = R_matrix.to(sh_complex.device)
    
    # Apply rotation: rotated = signal @ R_matrix.T
    # sh_complex is [n_samples, n_channels]
    # R_matrix is [n_channels, n_channels]
    sh_rotated_complex = torch.matmul(sh_complex, R_matrix.T)
    
    # Convert back to real SH
    sh_rotated_real = complex_to_real_ambisonics(sh_rotated_complex)
    
    return sh_rotated_real
