"""
I tried it with the DDIM sampler and it didn't work.

Probably need to use the k-diffusion sampler with it
from https://gist.githubusercontent.com/trygvebw/c71334dd127d537a15e9d59790f7f5e1/raw/a846393251f5be8289d4febc75a19f1f962aabcc/find_noise.py

needs https://github.com/crowsonkb/k-diffusion
"""

from contextlib import nullcontext
from typing import Iterable, TypeAlias, List

from imaginairy.api import load_model
from PIL import Image
import torch
from torch import autocast
from imaginairy.img_utils import pillow_img_to_model_latent
from imaginairy.vendored import k_diffusion as K
from imaginairy.samplers.base import get_sampler
from imaginairy.schema import ImagineResult
from imaginairy import ImaginePrompt
from einops import rearrange
import numpy as np
from dataclasses import dataclass
import abc
from enum import Enum

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


def find_noise_for_image(model, pil_img, prompt, steps=50, cond_scale=1.0, half=True):
    img_latent = pillow_img_to_model_latent(model, pil_img, batch_size=1, half=half)
    return find_noise_for_latent(
        model,
        img_latent,
        prompt,
        steps=steps,
        cond_scale=cond_scale,
    )


def find_noise_for_latent(model, img_latent, prompt, steps=50, cond_scale=1.0):
    x = img_latent

    _autocast = autocast if get_device() in ("cuda", "cpu") else nullcontext
    with torch.no_grad():
        with _autocast(get_device()):
            uncond = model.get_learned_conditioning([""])
            cond = model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    with torch.no_grad():
        with _autocast(get_device()):
            for i in range(1, len(sigmas)):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [
                    K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)
                ]
                t = dnw.sigma_to_t(sigma_in)

                eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

                d = (x - denoised) / sigmas[i]
                dt = sigmas[i] - sigmas[i - 1]

                x = x + d * dt

                del (
                    x_in,
                    sigma_in,
                    cond_in,
                    c_out,
                    c_in,
                    t,
                )
                del eps, denoised_uncond, denoised_cond, denoised, d, dt

            return x / x.std()


def from_noise(
        prompts,
        from_prompts=None,
        target_prompts=None,
        interpolation_percent=None,
        latent_channels=4,
        downsampling_factor=8,
        precision="autocast",
        ddim_eta=0.0,
        img_callback=None,
        half_mode=None,
        initial_noise_tensor=None,
        initial_text_cond=None,
        use_seq_weightning=False,
):
    model = load_model()

    # only run half-mode on cuda. run it by default
    half_mode = half_mode is None and get_device() == "cuda"
    if half_mode:
        model = model.half()
        # needed when model is in half mode, remove if not using half mode
        # torch.set_default_tensor_type(torch.HalfTensor)
    prompts = [ImaginePrompt(prompts)] if isinstance(prompts, str) else prompts
    prompts = [prompts] if isinstance(prompts, ImaginePrompt) else prompts
    _img_callback = None

    with torch.no_grad(), platform_appropriate_autocast(
            precision
    ), fix_torch_nn_layer_norm(), fix_torch_group_norm():
        for prompt in prompts:
            with ImageLoggingContext(
                    prompt=prompt,
                    model=model,
                    img_callback=img_callback,
            ):
                model.tile_mode(prompt.tile_mode)

                uc = None
                if prompt.prompt_strength != 1.0:
                    uc = model.get_learned_conditioning(1 * [""])
                    log_conditioning(uc, "neutral conditioning")

                if use_seq_weightning:
                    print("Prompts length: " + str(len(prompts)))
                    texts = [subprompt.text for subprompt in from_prompts] + [subprompt.text for subprompt in
                                                                              target_prompts]
                    cond_weights = [1.0 - interpolation_percent for _ in from_prompts] + [interpolation_percent for _ in
                                                                                          target_prompts]
                    cond_arities = [len(cond_weights)]
                    c = model.get_learned_conditioning(texts)

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
