# Copied from Stan's notebook - Super_resolution_and_inpainting_Stan.ipynb
# DO NOT CLEAN THIS CODE

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from interface import SatellitePlugin
from .models import U_net_generator, UNetSRGenerator


""" Helper functions for converting the (H, W, 12) TIFF images to (H, W, 3) RGB """

def normalize_band(band):
    """Normalize a band to 0-1 range for display.

    Uses 2-98 percentile stretch.
    NaN pixels are replaced with 0.
    """

    band = np.nan_to_num(band, nan=0.0)
    vmin, vmax = np.percentile(band[band > 0], [2, 98]) if (band > 0).any() else (0, 1)
    if vmax == vmin:
        return np.zeros_like(band)
    return np.clip((band - vmin) / (vmax - vmin), 0, 1)

def prepare_rgb(img_tif):
    """
    img_tif: (H, W, 12) float
    returns: (H, W, 3) uint8
    """
    rgb = np.stack([
        normalize_band(img_tif[:, :, 3]),  # R
        normalize_band(img_tif[:, :, 2]),  # G
        normalize_band(img_tif[:, :, 1]),  # B
    ], axis=-1)
    return (rgb * 255).astype(np.uint8)

def load_rgb_uint8(path):
    """Load RGB image as uint8 (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def rgb_to_tensor(rgb_uint8):

    """Convert uint8 RGB to float tensor (3, H, W) in [0,1]."""
    x = torch.from_numpy(rgb_uint8).float() / 255.0
    return x.permute(2, 0, 1)

def create_inpaint_mask(rgb_uint8, threshold=1):
    """
    Hole mask: (H, W) between {0,255} where 255 means hole.
    """
    hole = (
        (rgb_uint8[:, :, 0] <= threshold) &
        (rgb_uint8[:, :, 1] <= threshold) &
        (rgb_uint8[:, :, 2] <= threshold)
    )
    return hole.astype(np.uint8) * 255

def valid_pixel_mask_uint8(rgb_uint8, threshold=1):
    """
    Boolean where True means 'known pixel'.
    """
    return (
        (rgb_uint8[:, :, 0] > threshold) |
        (rgb_uint8[:, :, 1] > threshold) |
        (rgb_uint8[:, :, 2] > threshold)
    )


@torch.no_grad()
def infer_inpaint_with_mask(
    net,
    rgb_uint8: np.ndarray,     # image with holes
    hole_mask_u8: np.ndarray,  # 255 = hole
    device,
    corrupt_mode="zero", #real data has black holes
):
    net.eval()

    # input image
    x_obs = torch.from_numpy(rgb_uint8).float().permute(2,0,1) / 255.0

    # mask to {0,1}
    hm = hole_mask_u8.astype(np.float32)
    if hm.max() > 1.0:
        hm = hm / 255.0
    mask = torch.from_numpy(hm)[None].clamp(0,1)

    # fill hole area (zeros match corrupted output)
    if corrupt_mode == "zero":
        fill = torch.zeros_like(x_obs)
    elif corrupt_mode == "blur":
        fill = F.avg_pool2d(x_obs.unsqueeze(0), kernel_size=31, stride=1, padding=15)[0]
    else:
        raise ValueError("corrupt_mode must be 'blur' or 'zero'")

    x_corrupt = x_obs * (1.0 - mask) + fill * mask

    x_in = torch.cat([x_corrupt, mask], dim=0).unsqueeze(0).to(device)
    delta = net(x_in)[0].cpu()

    # apply prediction only inside hole
    hole_pred = (x_corrupt + delta * mask).clamp(0,1)

    # paste back into original image
    filled = x_obs * (1.0 - mask) + hole_pred * mask

    filled_u8 = (filled.permute(1,2,0).numpy() * 255).astype(np.uint8)
    pred_u8   = (hole_pred.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return filled_u8, pred_u8


PREFIX_INPAINT = "[Inpainting]"


class StanInpainter(SatellitePlugin):
    """Inpainting using Stable Diffusion pipeline."""

    def __init__(self):
        self._pipe = None
        self._device = None

    @property
    def name(self):
        return "Inpainting"

    def _load_pipeline(self):
        if self._pipe is None:
            from diffusers import StableDiffusionInpaintPipeline

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = Path(__file__).parent.parent.parent / "data" / "checkpoints" / "stable-diffusion-inpainting"
            self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
        return self._pipe

    def run(self, image: np.ndarray):
        """
        Inpaint black holes in the image using Stable Diffusion.
        image: (H, W, C) float32 array
        """
        # Convert to RGB uint8 if needed
        if image.ndim == 3 and image.shape[2] > 3:
            # Multi-channel (e.g., 12-band satellite) - convert to RGB
            rgb_uint8 = prepare_rgb(image)
        elif image.max() <= 1.0:
            rgb_uint8 = (image * 255).astype(np.uint8)
        else:
            rgb_uint8 = image.astype(np.uint8)

        # Ensure 3 channels
        if rgb_uint8.ndim == 2:
            rgb_uint8 = np.stack([rgb_uint8]*3, axis=-1)
        elif rgb_uint8.shape[2] == 1:
            rgb_uint8 = np.concatenate([rgb_uint8]*3, axis=-1)

        # Create mask for black holes
        mask_u8 = create_inpaint_mask(rgb_uint8, threshold=1)

        # If no holes, return original
        if mask_u8.mean() < 1e-3:
            return [
                (rgb_uint8.astype(np.float32) / 255.0, {"name": f"{PREFIX_INPAINT} Input", "rgb": True}, "image"),
                (rgb_uint8.astype(np.float32) / 255.0, {"name": f"{PREFIX_INPAINT} Output (no holes)", "rgb": True}, "image")
            ]

        # Load pipeline
        pipe = self._load_pipeline()

        # Convert to PIL
        img_pil = Image.fromarray(rgb_uint8)
        mask_pil = Image.fromarray(mask_u8, mode="L")

        # Resize to 256 for SD
        SD_SIZE = 256
        orig_size = img_pil.size
        img_256 = img_pil.resize((SD_SIZE, SD_SIZE), resample=Image.BICUBIC)
        mask_256 = mask_pil.resize((SD_SIZE, SD_SIZE), resample=Image.NEAREST)

        # Inpaint
        PROMPT = "satellite photo, realistic terrain, consistent lighting, natural textures"
        NEG_PROMPT = "blurry, oversmooth, repeating patterns, artifacts, cartoon"

        gen = torch.Generator(device=self._device).manual_seed(42)

        out_256 = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=img_256,
            mask_image=mask_256,
            guidance_scale=6,
            num_inference_steps=30,
            strength=1.0,
            generator=gen,
        ).images[0]

        # Resize back to original
        out_img = out_256.resize(orig_size, resample=Image.BICUBIC)
        out_array = np.array(out_img).astype(np.float32) / 255.0

        return [
            (rgb_uint8.astype(np.float32) / 255.0, {"name": f"{PREFIX_INPAINT} Input", "rgb": True}, "image"),
            (mask_u8.astype(np.float32) / 255.0, {"name": f"{PREFIX_INPAINT} Mask", "colormap": "gray"}, "image"),
            (out_array, {"name": f"{PREFIX_INPAINT} Output", "rgb": True}, "image"),
        ]


PREFIX_SR = "[Super-Resolution]"


class StanSuperRes(SatellitePlugin):
    """Super-Resolution using trained U-Net model."""

    def __init__(self):
        self._net = None
        self._device = None

    @property
    def name(self):
        return "Super-Resolution"

    def _load_model(self):
        if self._net is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Try to load checkpoint
            ckpt_path = Path(__file__).parent.parent.parent / "data" / "checkpoints" / "srgan_unet_final.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Super-Resolution checkpoint not found at {ckpt_path}. "
                    "Please download or train the model first."
                )

            self._net = UNetSRGenerator(base=32).to(self._device)
            ckpt = torch.load(ckpt_path, map_location=self._device)
            self._net.load_state_dict(ckpt["model_state"])
            self._net.eval()

        return self._net

    def run(self, image: np.ndarray):
        """
        Super-resolve the image using trained U-Net.

        The model was trained to enhance 4x bicubic-upsampled images.
        Input is treated as the low-resolution source and upsampled 4x.
        Model runs at 512x512 resolution.

        image: (H, W, C) float32 array
        """
        import torchvision.transforms.functional as TF

        # Convert to RGB uint8 if needed
        if image.ndim == 3 and image.shape[2] > 3:
            rgb_uint8 = prepare_rgb(image)
        elif image.max() <= 1.0:
            rgb_uint8 = (image * 255).astype(np.uint8)
        else:
            rgb_uint8 = image.astype(np.uint8)

        # Ensure 3 channels
        if rgb_uint8.ndim == 2:
            rgb_uint8 = np.stack([rgb_uint8]*3, axis=-1)

        # Load model
        net = self._load_model()

        # Convert to PIL - treat input as low-res image
        lr_pil = Image.fromarray(rgb_uint8)
        orig_h, orig_w = rgb_uint8.shape[:2]

        # Model was trained on 512x512, with 4x upscaling from 128x128
        # Upsample input 4x with bicubic, then resize to 512 for the model
        MODEL_SIZE = 512
        target_h, target_w = orig_h * 4, orig_w * 4

        # Bicubic upsample 4x
        up_pil = lr_pil.resize((target_w, target_h), Image.BICUBIC)

        # Resize to model size for inference
        up_512 = up_pil.resize((MODEL_SIZE, MODEL_SIZE), Image.BICUBIC)

        # To tensor
        up_tensor = TF.to_tensor(up_512).unsqueeze(0).to(self._device)

        # Run SR
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self._device == "cuda")):
                delta = net(up_tensor)
                sr_tensor = (up_tensor + delta).clamp(0, 1)

        # Convert back to PIL and resize to target 4x resolution
        sr_512 = Image.fromarray((sr_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        sr_pil = sr_512.resize((target_w, target_h), Image.BICUBIC)

        # Convert to numpy arrays
        sr_array = np.array(sr_pil).astype(np.float32) / 255.0
        up_array = np.array(up_pil).astype(np.float32) / 255.0
        lr_array = rgb_uint8.astype(np.float32) / 255.0

        return [
            (lr_array, {"name": f"{PREFIX_SR} Input ({orig_w}x{orig_h})", "rgb": True}, "image"),
            (up_array, {"name": f"{PREFIX_SR} Bicubic 4x ({target_w}x{target_h})", "rgb": True}, "image"),
            (sr_array, {"name": f"{PREFIX_SR} Output ({target_w}x{target_h})", "rgb": True}, "image"),
        ]
