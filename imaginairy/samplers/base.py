# pylama:ignore=W0613
import torch
from torch import nn, Tensor
from typing import Optional, Iterable, List

from imaginairy.utils import get_device

SAMPLER_TYPE_OPTIONS = [
    "plms",
    "ddim",
    "k_lms",
    "k_dpm_2",
    "k_dpm_2_a",
    "k_euler",
    "k_euler_a",
    "k_heun",
]

_k_sampler_type_lookup = {
    "k_dpm_2": "dpm_2",
    "k_dpm_2_a": "dpm_2_ancestral",
    "k_euler": "euler",
    "k_euler_a": "euler_ancestral",
    "k_heun": "heun",
    "k_lms": "lms",
}


def get_sampler(sampler_type, model):
    from imaginairy.samplers.ddim import DDIMSampler  # noqa
    from imaginairy.samplers.kdiff import KDiffusionSampler  # noqa
    from imaginairy.samplers.plms import PLMSSampler  # noqa

    sampler_type = sampler_type.lower()
    if sampler_type == "plms":
        return PLMSSampler(model)
    if sampler_type == "ddim":
        return DDIMSampler(model)
    if sampler_type.startswith("k_"):
        sampler_type = _k_sampler_type_lookup[sampler_type]
        return KDiffusionSampler(model, sampler_type)
    raise ValueError("invalid sampler_type")


class CFGDenoiser(nn.Module):
    """
    Conditional forward guidance wrapper
    """

    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask=None, orig_latent=None):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        denoised = uncond + (cond - uncond) * cond_scale

        if mask is not None:
            assert orig_latent is not None
            mask_inv = 1.0 - mask
            denoised = (orig_latent * mask_inv) + (mask * denoised)

        return denoised


class KCFGDenoiser(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(
            self,
            x: Tensor,
            sigma: Tensor,
            uncond: Tensor,
            cond: Tensor,
            cond_scale: float,
            cond_arities: Iterable[int],
            cond_weights: Optional[Iterable[float]]
    ) -> Tensor:
        uncond_count = uncond.size(dim=0)
        cond_count = cond.size(dim=0)
        cond_in = torch.cat((uncond, cond))
        del uncond, cond
        cond_arities_tensor = torch.tensor(cond_arities, device=cond_in.device)
        x_in = cat_self_with_repeat_interleaved(t=x, factors_tensor=cond_arities_tensor, factors=cond_arities,
                                                output_size=cond_count)
        del x
        sigma_in = cat_self_with_repeat_interleaved(t=sigma, factors_tensor=cond_arities_tensor, factors=cond_arities,
                                                    output_size=cond_count)
        del sigma

        uncond_out, conds_out = self.inner_model(x_in, sigma_in, cond=cond_in).split([uncond_count, cond_count])
        del x_in, sigma_in, cond_in
        unconds = repeat_interleave_along_dim_0(t=uncond_out, factors_tensor=cond_arities_tensor, factors=cond_arities,
                                                output_size=cond_count)
        del cond_arities_tensor
        # transform
        #   tensor([0.5, 0.1])
        # into:
        #   tensor([[[[0.5000]]],
        #           [[[0.1000]]]])
        weight_tensor = (
                torch.tensor(cond_weights, device=uncond_out.device, dtype=unconds.dtype) * cond_scale).reshape(
            len(cond_weights), 1, 1, 1)
        deltas: Tensor = (conds_out - unconds) * weight_tensor
        del conds_out, unconds, weight_tensor
        cond = sum_along_slices_of_dim_0(deltas, arities=cond_arities)
        del deltas
        return uncond_out + cond


def cat_self_with_repeat_interleaved(t: Tensor, factors: Iterable[int], factors_tensor: Tensor,
                                     output_size: int) -> Tensor:
    """
    Fast-paths for a pattern which in its worst-case looks like:
    t=torch.tensor([[0,1],[2,3]])
    factors=(2,3)
    torch.cat((t, t.repeat_interleave(factors, dim=0)))
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3]])
    Fast-path:
      `len(factors) == 1`
      it's just a normal repeat
    t=torch.tensor([[0,1]])
    factors=(2)
    tensor([[0, 1],
            [0, 1],
            [0, 1]])

    t=torch.tensor([[0,1],[2,3]])
    factors=(2)
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [2, 3],
            [0, 1],
            [2, 3]])
    """
    if len(factors) == 1:
        return repeat_along_dim_0(t, factors[0] + 1)
    return torch.cat((t, repeat_interleave_along_dim_0(t=t, factors_tensor=factors_tensor, factors=factors,
                                                       output_size=output_size)))


def repeat_interleave_along_dim_0(t: Tensor, factors: Iterable[int], factors_tensor: Tensor,
                                  output_size: int) -> Tensor:
    """
    repeat_interleave()s a tensor's contents along its 0th dim.
    factors=(2,3)
    factors_tensor = torch.tensor(factors)
    output_size=factors_tensor.sum().item() # 5
    t=torch.tensor([[0,1],[2,3]])
    repeat_interleave_along_dim_0(t=t, factors=factors, factors_tensor=factors_tensor, output_size=output_size)
    tensor([[0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3]])
    """
    factors_len = len(factors)
    assert factors_len >= 1
    if len(factors) == 1:
        # prefer repeat() whenever we can, because MPS doesn't support repeat_interleave()
        return repeat_along_dim_0(t, factors[0])
    if t.device.type != 'mps':
        return t.repeat_interleave(factors_tensor, dim=0, output_size=output_size)
    return torch.cat([repeat_along_dim_0(split, factor) for split, factor in zip(t.split(1, dim=0), factors)])


def repeat_along_dim_0(t: Tensor, factor: int) -> Tensor:
    """
    Repeats a tensor's contents along its 0th dim `factor` times.
    repeat_along_dim_0(torch.tensor([[0,1]]), 2)
    tensor([[0, 1],
            [0, 1]])
    # shape changes from (1, 2)
    #                 to (2, 2)

    repeat_along_dim_0(torch.tensor([[0,1],[2,3]]), 2)
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [2, 3]])
    # shape changes from (2, 2)
    #                 to (4, 2)
    """
    assert factor >= 1
    if factor == 1:
        return t
    if t.size(dim=0) == 1:
        # prefer expand() whenever we can, since doesn't copy
        return t.expand(factor * t.size(dim=0), *(-1,) * (t.ndim - 1))
    return t.repeat((factor, *(1,) * (t.ndim - 1)))


def sum_along_slices_of_dim_0(t: Tensor, arities: Iterable[int]) -> Tensor:
    """
    Implements fast-path for a pattern which in the worst-case looks like this:
    t=torch.tensor([[1],[2],[3]])
    arities=(2,1)
    torch.cat([torch.sum(split, dim=0, keepdim=True) for split in t.split(arities)])
    tensor([[3],
            [3]])
    Fast-path:
      `len(arities) == 1`
      it's just a normal sum(t, dim=0, keepdim=True)
    t=torch.tensor([[1],[2]])
    arities=(2)
    t.sum(dim=0, keepdim=True)
    tensor([[3]])
    """
    if len(arities) == 1:
        if t.size(dim=0) == 1:
            return t
        return t.sum(dim=0, keepdim=True)
    splits: List[Tensor] = t.split(arities)
    del t
    sums: List[Tensor] = [torch.sum(split, dim=0, keepdim=True) for split in splits]
    del splits
    return torch.cat(sums)


class DiffusionSampler:
    """
    wip

    hope to enforce an api upon samplers
    """

    def __init__(self, noise_prediction_model, sampler_func, device=get_device()):
        self.noise_prediction_model = noise_prediction_model
        self.cfg_noise_prediction_model = CFGDenoiser(noise_prediction_model)
        self.sampler_func = sampler_func
        self.device = device

    def zzsample(
            self,
            num_steps,
            text_conditioning,
            batch_size,
            shape,
            unconditional_guidance_scale,
            unconditional_conditioning,
            eta,
            initial_noise_tensor=None,
            img_callback=None,
    ):
        size = (batch_size, *shape)

        initial_noise_tensor = (
            torch.randn(size, device="cpu").to(get_device())
            if initial_noise_tensor is None
            else initial_noise_tensor
        )
        sigmas = self.noise_prediction_model.get_sigmas(num_steps)
        x = initial_noise_tensor * sigmas[0]

        samples = self.sampler_func(
            self.cfg_noise_prediction_model,
            x,
            sigmas,
            extra_args={
                "cond": text_conditioning,
                "uncond": unconditional_conditioning,
                "cond_scale": unconditional_guidance_scale,
            },
            disable=False,
        )

        return samples, None
