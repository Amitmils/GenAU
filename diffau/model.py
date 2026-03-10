import time
from math import ceil
import warnings

import torch
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from pathlib import Path

from diffau import sampling
from diffau.sdes import SDERegistry
from diffau.backbones import BackboneRegistry
from diffau.util.other import pad_spec,stft_sdr
from torch_pesq import PesqLoss


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=1, help="Number of files for upscaling performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="score_matching", help="The type of loss function to use.")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2", help="The weighting of the loss function.")
        parser.add_argument("--network_scaling", type=str, default=None, help="The type of loss scaling to use.")
        parser.add_argument("--c_in", type=str, default="1", help="The input scaling for x.")
        parser.add_argument("--c_out", type=str, default="1", help="The output scaling.")
        parser.add_argument("--c_skip", type=str, default="0", help="The skip connection scaling.")
        parser.add_argument("--sigma_data", type=float, default=0.1, help="The data standard deviation.")
        parser.add_argument("--l1_weight", type=float, default=0.001, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--sr", type=int, default=16000, help="The sample rate of the audio files.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='score_matching', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, input_order = 1,output_order = 3,
        first_low_order_channel = 0, l1_weight=0.001, sr=16000, data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        self.input_order = input_order
        self.output_order = output_order
        self.first_low_order_channel = first_low_order_channel
        self.num_low_order_channels = int((input_order+1)**2 - first_low_order_channel)
        assert self.num_low_order_channels > 0, "num of low order channels is 0!"
        self.num_high_order_channels = int((self.output_order+1)**2 -(input_order+1)**2)
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(self.num_low_order_channels,self.num_high_order_channels,**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.sr = sr

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0, output_order=self.output_order,first_low_order_channel=self.first_low_order_channel)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode = True, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, forward_out, x_t, z, t, mean, x_0, x_T):
        """
        Different loss functions can be used to train the score model, see the paper: 
        
        Julius Richter, Danilo de Oliveira, and Timo Gerkmann
        "Investigating Training Objectives for Generative Speech Enhancement"
        https://arxiv.org/abs/2409.10753
        """

        if self.sde.__class__.__name__.startswith('FlowMatching'):
            predicted_velocity = forward_out
            if self.loss_type == "velocity_loss":
                true_velocity = (x_0 - x_T)
                losses = torch.square(torch.abs(predicted_velocity - true_velocity)) 
                loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
            elif self.loss_type == "denoiser":
                x_hat = x_t + t[:, None, None, None] * predicted_velocity
                losses = torch.square(torch.abs(x_hat - x_0)) 
                if self.loss_weighting == "1":
                    losses = losses
                elif self.loss_weighting == "1-t^2":
                    losses = losses *(1 -  t[:, None, None, None])**2
                elif self.loss_weighting == "logt":
                    losses = losses * torch.log(2 - t[:, None, None, None])
                else:
                    raise ValueError("Invalid loss weighting for Flow Matching denoiser: {}".format(self.loss_weighting))
                
                # Primary MSE loss
                loss_mse = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)) / losses.shape[1] #normalize by channels for LR
                self._last_loss_mse = loss_mse.detach()
                loss = loss_mse

            else:
                raise ValueError("Invalid loss type for Flow Matching: {}".format(self.loss_type))
        else:
            sigma = self.sde._std(t)[:, None, None, None]
            if self.loss_type == "score_matching":
                score = forward_out
                if self.loss_weighting == "sigma^2":
                    losses = torch.square(torch.abs(score * sigma + z)) # Eq. (7)
                else:
                    raise ValueError("Invalid loss weighting for loss_type=score_matching: {}".format(self.loss_weighting))
                # Sum over spatial dimensions and channels and mean over batch
                loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)) / losses.shape[1] #normalize by channels

            elif self.loss_type == "denoiser":
                score = forward_out
                x_hat = score * sigma.pow(2) + x_t # equivalent to Eq. (10) to ensure no errors in backprop
                losses = torch.square(torch.abs(x_hat - x_0)) # Eq. (8)
                if self.loss_weighting == "1":
                    losses = losses
                elif self.loss_weighting == "sigma^2":
                    losses = losses * sigma**2
                elif self.loss_weighting == "edm":
                    losses = ((sigma**2 + self.sigma_data**2)/((sigma*self.sigma_data)**2))[:, None, None, None] * losses
                else:
                    raise ValueError("Invalid loss weighting for loss_type=denoiser: {}".format(self.loss_weighting))
                # Sum over spatial dimensions and channels and mean over batch
                loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))  / losses.shape[1] #normalize by channels
            else:
                raise ValueError("Invalid loss type: {}".format(self.loss_type))

        return loss

    def _step(self, batch, batch_idx):
        x = batch
        low_order_ambi_channels = x[:,:self.num_low_order_channels] #If we dont want to train on all the low order ambisonics channels
        x_0 = x[:,self.num_low_order_channels:self.num_low_order_channels + self.num_high_order_channels] #Precaution if we dont want to upscale to the full extent
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        if self.sde.__class__.__name__.startswith('FlowMatching'):
            x_T = self.sde.prior_sampling(x_0,low_order_ambi_channels).to(x.device)
            mean, std = self.sde.marginal_prob(x_0, x_T, t)
            x_t = mean
            z = None
        else:
            z = torch.randn_like(x_0)  # i.i.d. normal distributed with var=0.5
            mean, std = self.sde.marginal_prob(x_0, None, t)
            sigma = std[:, None, None, None]
            x_t = mean + sigma * z
            x_T = None
        forward_out = self(x_t,low_order_ambi_channels, t)
        loss = self._loss(forward_out, x_t, z, t, mean, x_0, x_T)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        
        # Log individual loss components if ICC loss is used
        if hasattr(self, '_last_loss_mse'):
            self.log('train_loss_mse', self._last_loss_mse, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train_loss_icc', self._last_loss_icc, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Evaluate
        if batch_idx == 0 and self.num_eval_files != 0:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1

            # Split the evaluation files among the GPUs
            eval_files_per_gpu = self.num_eval_files // world_size

            eval_files = self.data_module.valid_set.files[:self.num_eval_files]
            meta_data = self.data_module.valid_set.meta_data

            # Select the files for this GPU     
            if rank == world_size - 1:
                eval_files = eval_files[rank*eval_files_per_gpu:]
            else:   
                eval_files = eval_files[rank*eval_files_per_gpu:(rank+1)*eval_files_per_gpu]

            # Evaluate the performance of the model
            sdr_full_audio_grps = {}

            for file in eval_files:
                x, sr_x = load(file)
                id = int(Path(file).stem.split('_')[1])
                audio_info = dict(meta_data[meta_data.ID == id].iloc[0])
                x = x[:(self.output_order+1)**2] #truncate if more channels than output order
                y = x.clone()
                y = y[self.first_low_order_channel:(self.input_order+1)**2] 
                x_hat = self.upscale(y, N=self.sde.N) #Outputs the missing channels from order input to order output

                x_hoa_stft = self._stft(x[(self.input_order+1)**2:])
                x_hat_hoa_stft = self._stft(x_hat)

                sdr_full_audio = stft_sdr(x_hoa_stft, x_hat_hoa_stft, full_audio_sdr=True)[1]
                sdr_full_audio_grps.setdefault("avg",[]).append(sdr_full_audio)
                if audio_info.get('source',None) is not None:
                    sdr_full_audio_grps.setdefault(audio_info['source'],[]).append(sdr_full_audio)

            for metric_name,grped_vals in [('stft_sdr_full_audio',sdr_full_audio_grps)]:
                for grp,vals in grped_vals.items():
                    self.log(f"{metric_name}_{grp}",torch.stack(vals).mean().to(self.device),on_step=False, on_epoch=True, sync_dist=True)

        #if we are in dummy mode, we only want the eval of reverse, loss doesnt matter
        if self.data_module.dummy !=-1 and self.data_module.dummy<100:
            loss = torch.tensor(0, device=self.device)
        else:
            loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def forward(self, x_t, y, t):
        """
        The model forward pass. In [1] and [2], the model estimates the score function. In [3], the model estimates 
        either the score function or the target data for the Schrödinger bridge (loss_type='data_prediction').
        
        [1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, and  Timo Gerkmann 
            "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models"
            IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023. 

        [2] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann
            "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation"
            ISCA Interspecch, Kos, Greece, Sept. 2024. 

        [3] Julius Richter, Danilo de Oliveira, and Timo Gerkmann
            "Investigating Training Objectives for Generative Speech Enhancement"
            https://arxiv.org/abs/2409.10753

        """

        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            F = self.dnn(self._c_in(t) * x_t, self._c_in(t) * y, t)
            
            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]

            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * x_t + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (F - x_t) / sigmas.pow(2)
                return score
            elif self.loss_type == 'data_prediction':
                x_hat = self._c_skip(t) * x_t + self._c_out(t) * F
                return x_hat
                
        # In [1] and [2], we use the old code:
        else:
            dnn_input = torch.cat([y,x_t], dim=1) #low order channels + noisy high order channels
            if self.loss_type == 'data_prediction':
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (self.dnn(dnn_input, t) - x_t) / sigmas.pow(2)
            else:
                score = -self.dnn(dnn_input, t)
            return score

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_in type: {}".format(self.c_in))
    
    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None] 
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_out type: {}".format(self.c_out))

    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_skip type: {}".format(self.c_skip))

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs, 'num_high_order_channels' : self.num_high_order_channels}
        return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        return sampling.get_ode_sampler(sde, self, y=y, **kwargs)

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)

    def get_flow_matching_sampler(self, y, N=None, **kwargs):
        N = self.sde.N if N is None else N
        kwargs = {**kwargs, 'num_high_order_channels' : self.num_high_order_channels}
        return sampling.get_flow_matching_sampler(self.sde,self, y=y, N=N, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def upscale(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.to(self.device))), 0)
        Y = pad_spec(Y)

        if self.sde.__class__.__name__ == 'VESDE':
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(predictor, corrector, Y.to(self.device), N=N, 
                    corrector_steps=corrector_steps, snr=snr, intermediate=False,
                    **kwargs)
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y.to(self.device),N=N, **kwargs)
            else:
                raise ValueError("Invalid sampler type for SGMSE sampling: {}".format(sampler_type))
        elif self.sde.__class__.__name__.startswith('FlowMatching'):
            sampler = self.get_flow_matching_sampler(y=Y.to(self.device), N=N, **kwargs)
        else:
            raise ValueError("Invalid SDE type for speech enhancement: {}".format(self.sde.__class__.__name__))

        sample, nfe = sampler()
        x_hat = self.to_audio(sample, T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/self.sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
