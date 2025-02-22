import sys
sys.path.append("submodules/FlowEdit")

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
import os
from FlowEdit_utils import FlowEditSD3, FlowEditFLUX
from tqdm import tqdm

from typing import List
from PIL.Image import Image as PILImage
from contextlib import contextmanager


default_src_prompt = "Satellite image of an urban area with modern and older buildings, roads, green spaces, and a unique white angular structure. Some areas appear distorted, with blurring and warping artifacts near edges and trees."
default_tar_prompt = "Clear satellite image of an urban area with sharp buildings, smooth edges, and no distortions. Roads, green spaces, and the white angular structure are crisp, with natural lighting and well-defined textures."
negative_prompt = ""

def numpy_to_pil(numpy_img):
    # Ensure the array is uint8
    if numpy_img.dtype != np.uint8:
        numpy_img = (numpy_img * 255 + 0.5).clip(0, 255).astype(np.uint8)
    
    # Handle different number of channels
    if len(numpy_img.shape) == 2:
        # Grayscale
        return Image.fromarray(numpy_img, mode='L')
    elif len(numpy_img.shape) == 3:
        if numpy_img.shape[2] == 3:
            # RGB
            return Image.fromarray(numpy_img, mode='RGB')
        elif numpy_img.shape[2] == 4:
            # RGBA
            return Image.fromarray(numpy_img, mode='RGBA')
    
    raise ValueError("Unsupported array shape")

class FlowEditRefineIDU:
    def __init__(self, save_path, device="cuda:0", model_type="FLUX"):
        self.device = device
        self.save_path = save_path
        self.model_type = model_type
        if model_type == 'FLUX':
            # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16) 
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        elif model_type == 'SD3':
            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        self.scheduler = pipe.scheduler
        self.pipe = pipe.to(self.device)
        os.makedirs(save_path, exist_ok=True)
        print(f"Initialized FlowEdit with {model_type} model.")

    def __del__(self):
        if self.pipe is not None:
            try:
                self.pipe.to("cpu")  # Move to CPU before deleting
            except Exception as e:
                print(f"Error during model cleanup: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def model_loading_context(self):
        """Context manager for efficient model loading."""
        with torch.inference_mode():  # Use inference mode
            yield
    
    def run_single_image(self, img, src_prompt, tar_prompt, T_steps, n_avg, src_guidance_scale, tar_guidance_scale, n_min, n_max):
        img = numpy_to_pil(img)
        # img = Image.open("/project/jayinnn/mip-splatting/satellite_wild_op_loss_iterative/JAX_068_op_10_depth_0_sz_20_d20000_flux_v1/idu/e70_r300/render/00000.png")
        img = img.crop((0, 0, img.width - img.width % 16, img.height - img.height % 16))
        image_src = self.pipe.image_processor.preprocess(img)
        image_src = image_src.to(self.device).half()
        with torch.autocast("cuda"), torch.inference_mode():
            x0_src_denorm = self.pipe.vae.encode(image_src).latent_dist.mode()
        x0_src = (x0_src_denorm - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        # send to cuda
        x0_src = x0_src.to(self.device)

        flow_edit_func = FlowEditSD3 if self.model_type == 'SD3' else FlowEditFLUX
        if flow_edit_func is None: # Redundant, but good for clarity/future-proofing
            raise NotImplementedError(f"Sampler type {self.model_type} not implemented")

        # Perform FlowEdit
        x0_tar = flow_edit_func(self.pipe, self.scheduler, x0_src, src_prompt, tar_prompt,
                                "", T_steps, n_avg, src_guidance_scale, tar_guidance_scale,
                                n_min, n_max)

        # Decode the edited latent representation back to an image
        x0_tar_denorm = (x0_tar / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            image_tar = self.pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
        image_tar = self.pipe.image_processor.postprocess(image_tar)

        return image_tar[0]

    
    @torch.no_grad()
    def run(self, imgs: List[PILImage], src_prompt=default_src_prompt, tar_prompt=default_tar_prompt, T_steps=28, n_avg=1, src_guidance_scale=1.5, tar_guidance_scale=5.5, n_min=0, n_max=15):
        assert src_prompt is not None, "Should provide source prompt"
        assert tar_prompt is not None, "Should provide target prompt"
        refine_imgs = []
        for idx, img in enumerate(tqdm(imgs, desc=f"Refining images using FlowEdit with (min, max, avg) = ({n_min}, {n_max}, {n_avg})")):
            refine_img = self.run_single_image(img, src_prompt, tar_prompt, T_steps, n_avg, src_guidance_scale, tar_guidance_scale, n_min, n_max)
            refine_img.save(os.path.join(self.save_path, '{0:05d}'.format(idx) + ".png"))
            refine_imgs.append(refine_img)

        return refine_imgs