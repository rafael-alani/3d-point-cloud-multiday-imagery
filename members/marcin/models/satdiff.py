"""SD inpainting with ControlNet and tiled MultiDiffusion for large images."""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

# Local checkpoint paths (fallback to HuggingFace if not present)
_CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "checkpoints"
_LOCAL_SD_INPAINT = _CHECKPOINTS_DIR / "stable-diffusion-inpainting"
_LOCAL_CONTROLNET = _CHECKPOINTS_DIR / "sd-controlnet-canny"

# Use local if available, otherwise download from HuggingFace
SD_INPAINT_PATH = str(_LOCAL_SD_INPAINT) if _LOCAL_SD_INPAINT.exists() else "sd-legacy/stable-diffusion-inpainting"
CONTROLNET_CANNY_PATH = str(_LOCAL_CONTROLNET) if _LOCAL_CONTROLNET.exists() else "lllyasviel/sd-controlnet-canny"

RESTORATION_DEFAULTS = {
    "num_steps": 30,
    "strength": 0.9,
    "use_controlnet": True,
    "canny_low": 50,
    "canny_high": 150,
    "cn_scale": 0.8,
    "control_guidance_end": 0.6,
    "prompt": "satellite image, natural landscape, seamless texture",
}

STITCHING_DEFAULTS = {
    "num_steps": 30,
    "strength": 0.9,
    "use_controlnet": True,
    "canny_low": 50,
    "canny_high": 150,
    "cn_scale": 0.6,
    "control_guidance_end": 0.6,
    "prompt": "satellite image, seamless transition, continuous texture",
}

ENHANCEMENT_DEFAULTS = {
    "num_steps": 25,
    "strength": 0.4,
    "use_controlnet": False,
    "canny_low": 50,
    "canny_high": 150,
    "cn_scale": 1.2,
    "control_guidance_end": 0.9,
    "prompt": "satellite image, high detail, sharp texture",
}


def gaussian_weights(height, width, channels=4):
    """2D gaussian for tile blending."""
    sigma_h, sigma_w = height / 6, width / 6
    y = torch.arange(height, dtype=torch.float32) - (height - 1) / 2
    x = torch.arange(width, dtype=torch.float32) - (width - 1) / 2
    gauss = torch.outer(torch.exp(-y**2 / (2 * sigma_h**2)), torch.exp(-x**2 / (2 * sigma_w**2)))
    gauss = torch.clamp(gauss / gauss.max(), min=0.01)
    return gauss.unsqueeze(0).unsqueeze(0).expand(1, channels, -1, -1)


def get_tiles(height, width, tile_size=64, stride=32):
    """Get overlapping tile coordinates in latent space."""
    if height <= tile_size and width <= tile_size:
        return [(0, height, 0, width)]

    tiles = []
    for i in range((height - tile_size + stride) // stride + 1):
        for j in range((width - tile_size + stride) // stride + 1):
            y_start = min(i * stride, max(0, height - tile_size))
            x_start = min(j * stride, max(0, width - tile_size))
            tiles.append((y_start, min(y_start + tile_size, height),
                         x_start, min(x_start + tile_size, width)))
    return list(set(tiles))


class SatDiffEngine:
    """SD inpainting + optional ControlNet, with MultiDiffusion for big images."""

    def __init__(self):
        self.pipeline = None
        self.pipeline_controlnet = None

    def load_pipeline(self, use_controlnet):
        # DYNAMICALLY loads different diffusion pipelines (cn vs no cn)
        if use_controlnet and self.pipeline_controlnet:
            return self.pipeline_controlnet
        if not use_controlnet and self.pipeline:
            return self.pipeline

        dtype = torch.float16

        if use_controlnet:
            print(f"Loading SD + ControlNet from {SD_INPAINT_PATH}...")
            controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_PATH, torch_dtype=dtype)
            self.pipeline_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                SD_INPAINT_PATH,
                controlnet=controlnet, torch_dtype=dtype, safety_checker=None
            ).to("cuda")
            self.pipeline_controlnet.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline_controlnet.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
            )
            print("Loaded")
            return self.pipeline_controlnet
        else:
            print(f"Loading SD inpainting from {SD_INPAINT_PATH}...")
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                SD_INPAINT_PATH,
                torch_dtype=dtype, safety_checker=None
            ).to("cuda")
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
            )
            print("Loaded")
            return self.pipeline

    def make_canny(self, image, mask, low, high):
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        # zero out edges inside mask
        dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        edges[dilated == 255] = 0
        return edges

    def prefill(self, image, mask):
        """Telea inpaint to give VAE a better starting point."""
        if not (mask == 255).any():
            return image.copy()
        image_uint8 = (image * 255).astype(np.uint8)
        filled = cv2.inpaint(image_uint8, mask, 5, cv2.INPAINT_TELEA)
        return filled.astype(np.float32) / 255

    @torch.no_grad()
    def tiled_inpaint(self, pipe, image_pil, mask_pil, control_image, height, width,
                      num_steps, strength, cn_scale, control_guidance_end, prompt, overlap=16):
        """MultiDiffusion: process tiles with gaussian blending."""
        device, dtype = pipe.device, pipe.unet.dtype

        # align to 8px
        aligned_height = (height // 8) * 8
        aligned_width = (width // 8) * 8
        latent_height, latent_width = aligned_height // 8, aligned_width // 8
        tile_size, stride = 64, 64 - overlap # 64 = native sd inpainting latent res

        # encode prompt
        prompt_embeds, _ = pipe.encode_prompt(prompt, device, 1, do_classifier_free_guidance=False)

        # prep tensors
        image_tensor = pipe.image_processor.preprocess(
            image_pil.resize((aligned_width, aligned_height), Image.LANCZOS)
        ).to(device, dtype)
        mask_tensor = pipe.mask_processor.preprocess(
            mask_pil.resize((aligned_width, aligned_height), Image.NEAREST)
        ).to(device, dtype)

        init_latents = pipe.vae.encode(image_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        mask_latents = torch.nn.functional.interpolate(mask_tensor, (latent_height, latent_width), mode="nearest")
        masked_latents = pipe.vae.encode(image_tensor * (1 - mask_tensor)).latent_dist.sample() * pipe.vae.config.scaling_factor

        # scheduler setup
        pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        start_step = max(num_steps - int(num_steps * strength), 0)
        timesteps = timesteps[start_step:]

        latents = pipe.scheduler.add_noise(init_latents, torch.randn_like(init_latents), timesteps[0:1])
        tile_weights = gaussian_weights(tile_size, tile_size, 4).to(device, dtype)

        # only process tiles that touch the mask
        all_tiles = get_tiles(latent_height, latent_width, tile_size, stride)
        active_tiles = [t for t in all_tiles if mask_latents[:, :, t[0]:t[1], t[2]:t[3]].sum() > 0]
        print(f"Processing {len(active_tiles)}/{len(all_tiles)} tiles for {aligned_width}x{aligned_height}...")

        for step_idx, timestep in enumerate(timesteps):
            noise_buffer = torch.zeros_like(latents)
            weight_buffer = torch.zeros_like(latents)

            for y_start, y_end, x_start, x_end in active_tiles:
                tile_latents = latents[:, :, y_start:y_end, x_start:x_end]
                tile_mask = mask_latents[:, :, y_start:y_end, x_start:x_end]
                tile_masked = masked_latents[:, :, y_start:y_end, x_start:x_end]

                model_input = pipe.scheduler.scale_model_input(tile_latents, timestep)
                model_input = torch.cat([model_input, tile_mask, tile_masked], dim=1)

                # controlnet
                controlnet_kwargs = {}
                if control_image and pipe == self.pipeline_controlnet:
                    crop = control_image.crop((x_start*8, y_start*8, x_end*8, y_end*8))
                    control_cond = pipe.prepare_control_image(
                        image=crop, width=tile_size*8, height=tile_size*8,
                        batch_size=1, num_images_per_prompt=1,
                        device=device, dtype=pipe.controlnet.dtype,
                        do_classifier_free_guidance=False,
                        crops_coords=None, resize_mode="default"
                    )
                    scale = cn_scale if step_idx / num_steps <= control_guidance_end else 0
                    down_residuals, mid_residual = pipe.controlnet(
                        pipe.scheduler.scale_model_input(tile_latents, timestep), timestep,
                        encoder_hidden_states=prompt_embeds, controlnet_cond=control_cond,
                        conditioning_scale=scale, return_dict=False
                    )
                    controlnet_kwargs = {
                        "down_block_additional_residuals": down_residuals,
                        "mid_block_additional_residual": mid_residual
                    }

                noise_pred = pipe.unet(model_input, timestep, encoder_hidden_states=prompt_embeds, **controlnet_kwargs).sample
                noise_buffer[:, :, y_start:y_end, x_start:x_end] += noise_pred * tile_weights
                weight_buffer[:, :, y_start:y_end, x_start:x_end] += tile_weights

            # blend predictions
            has_weight = (weight_buffer > 0).to(dtype)
            blended_noise = torch.where(weight_buffer > 0, noise_buffer / weight_buffer, torch.zeros_like(noise_buffer))

            stepped_latents = pipe.scheduler.step(blended_noise, timestep, latents).prev_sample
            latents = has_weight * stepped_latents + (1 - has_weight) * latents

            # keep unmasked regions locked to original
            if step_idx < len(timesteps) - 1:
                next_timestep = timesteps[step_idx + 1 : step_idx + 2]
                original_noised = pipe.scheduler.add_noise(init_latents, torch.randn_like(init_latents), next_timestep)
                latents = (1 - mask_latents) * original_noised + mask_latents * latents

        # decode
        output = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        return pipe.image_processor.postprocess(output, output_type="pil")[0]

    def run(self, image, mask, num_steps, strength, use_controlnet, canny_low, canny_high,
            cn_scale, control_guidance_end, prompt, **_):
        pipe = self.load_pipeline(use_controlnet)
        height, width = image.shape[:2]
        aligned_height = (height // 8) * 8
        aligned_width = (width // 8) * 8

        filled_image = self.prefill(image, mask)
        image_pil = Image.fromarray((filled_image * 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask)

        control_image = None
        if use_controlnet:
            control_image = Image.fromarray(self.make_canny(filled_image, mask, canny_low, canny_high)).convert("RGB")

        # use tiled for large, direct for small
        if aligned_height > 512 or aligned_width > 512:
            result = self.tiled_inpaint(pipe, image_pil, mask_pil, control_image,
                                        aligned_height, aligned_width, num_steps, strength,
                                        cn_scale, control_guidance_end, prompt)
        else:
            image_pil = image_pil.resize((aligned_width, aligned_height), Image.LANCZOS)
            mask_pil = mask_pil.resize((aligned_width, aligned_height), Image.NEAREST)

            # setup configuration
            kwargs = {
                "prompt": prompt, "image": image_pil, "mask_image": mask_pil,
                "height": aligned_height, "width": aligned_width,
                "num_inference_steps": num_steps, "guidance_scale": 1.0, "strength": strength,
            }
            if use_controlnet and control_image:
                control_image = control_image.resize((aligned_width, aligned_height), Image.LANCZOS)
                kwargs["control_image"] = control_image
                kwargs["controlnet_conditioning_scale"] = cn_scale
                kwargs["control_guidance_end"] = control_guidance_end
            result = pipe(**kwargs).images[0]

        if (aligned_height, aligned_width) != (height, width):
            result = result.resize((width, height), Image.LANCZOS)

        return np.array(result).astype(np.float32) / 255

    def restore(self, image, mask, **config):
        return self.run(image, mask, **{**RESTORATION_DEFAULTS, **config})

    def stitch(self, composite, gap_mask, **config):
        return self.run(composite, gap_mask, **{**STITCHING_DEFAULTS, **config})

    def enhance(self, image, mask, **config):
        return self.run(image, mask, **{**ENHANCEMENT_DEFAULTS, **config})
