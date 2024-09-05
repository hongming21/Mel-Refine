# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch utilities: Utilities related to PyTorch
"""

from typing import List, Optional, Tuple, Union

from . import logging
from .import_utils import is_torch_available, is_torch_version


if is_torch_available():
    import torch
    from torch.fft import fftn, fftshift, ifftn, ifftshift

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
except (ImportError, ModuleNotFoundError):

    def maybe_allow_in_graph(cls):
        return cls


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def is_compiled_module(module) -> bool:
    """Check whether the module was compiled with torch.compile()"""
    if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def fourier_filter(x_in: "torch.Tensor", threshold: str, scale: int) -> "torch.Tensor":
    """Fourier filter as introduced in FreeU (https://arxiv.org/abs/2309.11497).

    This version of the method comes from here:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706

    Args:
        x_in: Input tensor.
        threshold: Either 'high' or 'low' to determine the filtering region.
        scale: The scale applied to the filtered frequency components.
    
    Returns:
        x_filtered: Filtered tensor in the spatial domain.
    """
    x = x_in
    B, C, H, W = x.shape

    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = torch.fft.fftn(x, dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    mask = torch.ones((B, C, H, W), device=x.device, dtype=torch.float32)
    bool_mask = torch.ones_like(mask, dtype=torch.bool)
    crow, ccol = H // 2, W // 2
    r_threshold = H // 4
    c_threshold = W//4 if W//4 > 1 else 1

    # Apply the threshold condition
    if threshold == "high":
        # Center region is set to False (will not be scaled)
        bool_mask[..., crow - r_threshold : crow + r_threshold, ccol - c_threshold : ccol + c_threshold] = False
    elif threshold == "low":
        # Invert the mask to select regions outside the center
        bool_mask[..., crow - r_threshold : crow + r_threshold, ccol - c_threshold : ccol + c_threshold] = False
        bool_mask = ~bool_mask  # Invert the mask to set outer regions to False

    mask[bool_mask] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)



def apply_freeu(
    resolution_idx: int, hidden_states: "torch.Tensor", res_hidden_states: "torch.Tensor", **freeu_kwargs
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Applies the FreeU mechanism as introduced in https:
    //arxiv.org/abs/2309.11497. Adapted from the official code repository: https://github.com/ChenyangSi/FreeU.

    Args:
        resolution_idx (`int`): Integer denoting the UNet block where FreeU is being applied.
        hidden_states (`torch.Tensor`): Inputs to the underlying block.
        res_hidden_states (`torch.Tensor`): Features from the skip block corresponding to the underlying block.
        s1 (`float`): Scaling factor for stage 1 to attenuate the contributions of the skip features.
        s2 (`float`): Scaling factor for stage 2 to attenuate the contributions of the skip features.
        b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
        b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
    """
    if resolution_idx == 0:
        hidden_mean = hidden_states.mean(1).unsqueeze(1)
        B = hidden_mean.shape[0]
        hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
        hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
        hidden_states = hidden_states * ((freeu_kwargs["m"]-1) * hidden_mean + 1)
        hidden_states=fourier_filter(hidden_states,threshold="high",scale=freeu_kwargs["b1"])
        res_hidden_states = fourier_filter(res_hidden_states, threshold="high", scale=freeu_kwargs["s1"])
    if resolution_idx == 1:
        hidden_mean = hidden_states.mean(1).unsqueeze(1)
        B = hidden_mean.shape[0]
        hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
        hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
        hidden_states = hidden_states * ((freeu_kwargs["m"]-1) * hidden_mean + 1)
        hidden_states=fourier_filter(hidden_states,threshold="high",scale=freeu_kwargs["b2"])
        res_hidden_states = fourier_filter(res_hidden_states, threshold="high", scale=freeu_kwargs["s2"])

    return hidden_states, res_hidden_states
