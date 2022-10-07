"""
I tried it with the DDIM sampler and it didn't work.

Probably need to use the k-diffusion sampler with it
from https://gist.githubusercontent.com/trygvebw/c71334dd127d537a15e9d59790f7f5e1/raw/a846393251f5be8289d4febc75a19f1f962aabcc/find_noise.py

needs https://github.com/crowsonkb/k-diffusion
"""

from contextlib import nullcontext
from typing import Iterable, Optional

from imaginairy.api import load_model
from PIL import Image
import torch
from torch import autocast, Tensor
from imaginairy.img_utils import pillow_img_to_model_latent
from imaginairy.vendored import k_diffusion as K
from imaginairy.samplers.base import get_sampler, cat_self_with_repeat_interleaved, repeat_interleave_along_dim_0, \
    sum_along_slices_of_dim_0
from imaginairy.schema import ImagineResult
from imaginairy import ImaginePrompt
from einops import rearrange
import numpy as np

from imaginairy.utils import (
    fix_torch_group_norm,
    fix_torch_nn_layer_norm,
    get_device,
    platform_appropriate_autocast,
)

from imaginairy.img_log import (
    ImageLoggingContext,
    log_conditioning,
)


def find_noise_for_image(model, pil_img, prompts: Iterable[str],
                         negative_prompts: Iterable[str], cond_weights: Optional[Iterable[float]],
                         cond_arities: Iterable[int], steps=50, cond_scale=1.0, half=True):
    img_latent = pillow_img_to_model_latent(model, pil_img, batch_size=1, half=half)
    return find_noise_for_latent(
        model,
        img_latent,
        prompts,
        cond_weights,
        cond_arities,
        negative_prompts,
        steps=steps,
        cond_scale=cond_scale,
    )


def find_noise_for_latent(model, img_latent, prompts: Iterable[str],
                          cond_weights: Optional[Iterable[float]],
                          cond_arities: Iterable[int],
                          negative_prompts: Iterable[str] = 1 * [""],
                          steps=50, cond_scale=1.0):
    x = img_latent

    _autocast = autocast if get_device() in ("cuda", "cpu") else nullcontext
    with torch.no_grad():
        with _autocast(get_device()):
            uncond = model.get_learned_conditioning(negative_prompts)
            cond = model.get_learned_conditioning(prompts)

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    with torch.no_grad():
        with _autocast(get_device()):
            for i in range(1, len(sigmas)):
                cond_arities_tensor = torch.tensor(cond_arities, device=cond.device)
                cond_count = cond.size(dim=0)
                uncond_count = uncond.size(dim=0)
                x_in = cat_self_with_repeat_interleaved(t=x,
                                                        factors_tensor=cond_arities_tensor,
                                                        factors=cond_arities,
                                                        output_size=cond_count)
                sigma_in = cat_self_with_repeat_interleaved(t=sigmas[i] * s_in,
                                                            factors_tensor=cond_arities_tensor,
                                                            factors=cond_arities,
                                                            output_size=cond_count)
                cond_in = torch.cat((uncond, cond))

                c_out, c_in = [
                    K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)
                ]

                t = dnw.sigma_to_t(sigma_in)

                eps = model.apply_model(x_in * c_in, t, cond=cond_in)

                uncond_out, conds_out = (x_in + eps * c_out).split([uncond_count, cond_count])
                del eps, c_out, x_in

                unconds = repeat_interleave_along_dim_0(t=uncond_out, factors_tensor=cond_arities_tensor,
                                                        factors=cond_arities,
                                                        output_size=cond_count)

                weight_tensor = (
                        torch.tensor(cond_weights, device=uncond_out.device, dtype=unconds.dtype) * cond_scale).reshape(
                    len(cond_weights), 1, 1, 1)
                deltas: Tensor = (conds_out - unconds) * weight_tensor
                del conds_out, unconds, weight_tensor
                conda = sum_along_slices_of_dim_0(deltas, arities=cond_arities)
                del deltas

                denoised = uncond_out + conda
                del uncond_out, conda

                d = (x - denoised) / sigmas[i]
                dt = sigmas[i] - sigmas[i - 1]

                x = x + d * dt

                del (
                    sigma_in,
                    cond_in,
                    c_in,
                    t,
                )
                del denoised, d, dt

            return x / x.std()


def from_noise(
        requests,
        prompts,
        negative_prompts=1 * [""],
        initial_noise_tensor=None,
        initial_text_cond=None,
        cond_weights=None,
        use_seq_weightning=False,
        latent_channels=4,
        downsampling_factor=8,
        precision="autocast",
        ddim_eta=0.0,
        half_mode=None,
        img_callback=None,
):
    model = load_model()

    # only run half-mode on cuda. run it by default
    half_mode = half_mode is None and get_device() == "cuda"
    if half_mode:
        model = model.half()
        # needed when model is in half mode, remove if not using half mode
        # torch.set_default_tensor_type(torch.HalfTensor)
    requests = [ImaginePrompt(requests)] if isinstance(requests, str) else requests
    requests = [requests] if isinstance(requests, ImaginePrompt) else requests
    _img_callback = None

    with torch.no_grad(), platform_appropriate_autocast(
            precision
    ), fix_torch_nn_layer_norm(), fix_torch_group_norm():
        for prompt in requests:
            with ImageLoggingContext(
                    prompt=prompt,
                    model=model,
                    img_callback=img_callback,
            ):
                model.tile_mode(prompt.tile_mode)

                uc = None
                if prompt.prompt_strength != 1.0:
                    uc = model.get_learned_conditioning(negative_prompts)
                    log_conditioning(uc, "neutral conditioning")

                if use_seq_weightning:
                    cond_arities = [len(cond_weights)]
                    c = model.get_learned_conditioning(prompts)

                elif prompt.conditioning is not None:
                    c = prompt.conditioning
                else:
                    total_weight = sum(wp.weight for wp in prompt.prompts)
                    c = sum(
                        model.get_learned_conditioning(wp.text)
                        * (wp.weight / total_weight)
                        for wp in prompt.prompts
                    )
                log_conditioning(c, "positive conditioning")

                shape = [
                    latent_channels,
                    prompt.height // downsampling_factor,
                    prompt.width // downsampling_factor,
                ]

                sampler_type = prompt.sampler_type

                sampler = get_sampler(sampler_type, model)

                if initial_text_cond is not None:
                    c = initial_text_cond

                if use_seq_weightning:
                    samples = sampler.sample(
                        initial_noise_tensor=initial_noise_tensor,
                        num_steps=prompt.steps,
                        conditioning=c,
                        batch_size=1,
                        shape=shape,
                        unconditional_guidance_scale=prompt.prompt_strength,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        img_callback=_img_callback,
                        use_seq_weightning=True,
                        cond_arities=cond_arities,
                        cond_weights=cond_weights
                    )
                else:
                    samples = sampler.sample(
                        initial_noise_tensor=initial_noise_tensor,
                        num_steps=prompt.steps,
                        conditioning=c,
                        batch_size=1,
                        shape=shape,
                        unconditional_guidance_scale=prompt.prompt_strength,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        img_callback=_img_callback,
                    )

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = x_sample.to(torch.float32)
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    x_sample_8_orig = x_sample.astype(np.uint8)
                    img = Image.fromarray(x_sample_8_orig)

                    yield c, ImagineResult(
                        img=img,
                        prompt=prompt,
                        upscaled_img=None,
                        is_nsfw=False,
                    )


if __name__ == "__main__":
    pass
