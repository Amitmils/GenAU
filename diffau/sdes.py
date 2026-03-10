"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""

import abc
import warnings

import numpy as np
from diffau.util.tensors import batch_broadcast
import torch

from diffau.util.registry import Registry


SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, y, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, y, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, y, t, stepsize):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = stepsize
        drift, diffusion = self.sde(x, y, t)
        f = drift * dt
        G = diffusion * torch.sqrt(dt)
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, y, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, y, t, *args)
                total_drift, diffusion = (
                    rsde_parts["total_drift"],
                    rsde_parts["diffusion"],
                )
                return total_drift, diffusion

            def rsde_parts(self, x, y, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, y, t, *args)
                score = score_model(x, y, t, *args)
                score_drift = (
                    -sde_diffusion[:, None, None, None] ** 2
                    * score
                    * (0.5 if self.probability_flow else 1.0)
                )
                diffusion = (
                    torch.zeros_like(sde_diffusion)
                    if self.probability_flow
                    else sde_diffusion
                )
                total_drift = sde_drift + score_drift
                return {
                    "total_drift": total_drift,
                    "diffusion": diffusion,
                    "sde_drift": sde_drift,
                    "sde_diffusion": sde_diffusion,
                    "score_drift": score_drift,
                    "score": score,
                }

            def discretize(self, x, y, t, stepsize):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(
                    x, 0, t, stepsize
                )  # the y here is 0 since thats our best guess
                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, y, t) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G


        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass

@SDERegistry.register("ve")
class VESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--sigma-min",
            type=float,
            default=0.05,
            help="The minimum sigma to use. 0.05 by default.",
        )
        parser.add_argument(
            "--sigma-max",
            type=float,
            default=1,
            help="The maximum sigma to use. 0.5 by default.",
        )
        parser.add_argument(
            "--N",
            type=int,
            default=30,
            help="The number of timesteps in the SDE discretization. 30 by default",
        )
        parser.add_argument(
            "--sampler_type",
            type=str,
            default="pc",
            help="Type of sampler to use. 'pc' by default.",
        )
        return parser
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, sampler_type='pc',**ignored_kwargs):
        """Construct a Variance Exploding SDE.

        dx = -sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.sampler_type = sampler_type


        # self.discrete_sigmas = torch.exp(
        #     np.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        # )
        self.N = N


    def copy(self):
        return VESDE(
            self.sigma_min,
            self.sigma_max,
            N=self.N,
            sampler_type=self.sampler_type,
        )

    @property
    def T(self):
        return 1

    def sde(self, x, y, t):
        # y place holder to be compatible with OUVE (TODO remove it entirely since its 0 there as well)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        drift = torch.zeros_like(x)
        return drift, diffusion

    def _std(self, t,exact=True):
        sigma_min, logsig = self.sigma_min, self.logsig

        #They should be more or less the same
        if exact:
            return sigma_min * torch.sqrt(torch.exp(2 * logsig * t) - 1)
        else:
            return sigma_min * (self.sigma_max / self.sigma_min) ** t

    def marginal_prob(self, x, y, t):
        # y place holder to be compatible with OUVE
        mean = x
        return mean, self._std(t)

    def prior_sampling(self, shape, y):
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        return torch.randn_like(y) * std[:, None, None, None]

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for VE SDE not yet implemented yet!")

@SDERegistry.register("flow_matching")
class FlowMatching(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--N",
            type=int,
            default=5,
            help="The number of timesteps in the SDE discretization. 50 by default",
        )
        parser.add_argument("--sampler_type", type=str, default="ode")
        return parser

    def __init__(self, sigma_min=1e-4, N=50, **kwargs):
        super().__init__(N)
        self.sigma_min = sigma_min  # Small noise for numerical stability
        

    @property
    def T(self):
        return 1
    
    def copy(self):
        return FlowMatching(sigma_min=self.sigma_min, N=self.N)

    def sde(self, x, y, t):
        # Flow matching uses straight paths: x_t = (1-t)*x_0 + t*x_1 + sigma*noise
        # Drift is just the velocity field
        drift = torch.zeros_like(x)  # Will be replaced by learned velocity
        diffusion = torch.full((x.shape[0],), self.sigma_min, device=x.device)
        return drift, diffusion
    
    def marginal_prob(self, x0, x_T, t):
        # Linear interpolation path
        mean = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x_T
        std = torch.full_like(t, self.sigma_min)
        return mean, std

    def prior_sampling(self,x_0,y):
        return torch.randn_like(x_0) 
    
    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for flow matching not yet implemented!")
