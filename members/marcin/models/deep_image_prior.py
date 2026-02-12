"""Deep Image Prior engine for unsupervised inpainting."""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim


# Default configurations for different tasks
DIP_RESTORATION_DEFAULTS = {
    "num_iter": 800,
    "lr": 0.01,
}

DIP_STITCHING_DEFAULTS = {
    "num_iter": 1000,  # More iterations for stitching
    "lr": 0.01,
}

DIP_ENHANCEMENT_DEFAULTS = {
    "num_iter": 800,
    "lr": 0.005,
}


class UNet(nn.Module):
    """Simple encoder-decoder network with skip connections."""

    def __init__(self, input_channels, output_channels):
        super().__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Decode with skip connections
        d3 = self.decoder3(e3)
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))

        return self.output(d1)


class DIPEngine:
    """Deep Image Prior engine for restoration and stitching.

    Uses untrained CNN as implicit regularizer for image restoration.
    """

    def _run_dip(
        self, image: np.ndarray, mask: np.ndarray, num_iter: int, lr: float, **_
    ) -> np.ndarray:
        """Core DIP optimization loop."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        height, width, channels = image.shape
        original_size = (width, height)

        # Downscale large images for memory
        MAX_SIZE = 512
        scale = 1.0

        if max(height, width) > MAX_SIZE:
            scale = MAX_SIZE / max(height, width)
            height = int(height * scale)
            width = int(width * scale)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Prepare tensors
        image_tensor = torch.from_numpy(image.astype(np.float32))
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Check if this is full-image enhancement (all mask values are 255)
        is_enhancement = np.all(mask == 255)

        if is_enhancement:
            # For enhancement: train on all pixels
            mask_tensor = torch.ones(1, 1, height, width, device=device)
        else:
            # For inpainting: only optimize on known pixels (where mask is 0)
            mask_tensor = torch.from_numpy((mask == 0).astype(np.float32))
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)

        # Random noise input
        noise = torch.randn(1, 32, height, width, device=device) * 0.1

        # Build and train network
        network = UNet(input_channels=32, output_channels=channels).to(device)
        optimizer = optim.Adam(network.parameters(), lr=lr)

        print(f"Running DIP for {num_iter} iterations...")
        for i in range(num_iter):
            optimizer.zero_grad()

            output = network(noise)
            loss = nn.functional.mse_loss(output, image_tensor, reduction="none")
            loss = (loss * mask_tensor).mean()

            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                print(f"  Iteration {i + 1}/{num_iter}, loss: {loss.item():.6f}")

        # Extract result
        with torch.no_grad():
            result = network(noise)
            result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()

        result = np.clip(result, 0, 1)

        # Upscale back to original size
        if scale < 1.0:
            result = cv2.resize(result, original_size, interpolation=cv2.INTER_LANCZOS4)

        print("DIP completed")
        return result

    def restore(
        self, image: np.ndarray, mask: np.ndarray, **config
    ) -> np.ndarray:
        """Restore masked regions of the image."""
        config = {**DIP_RESTORATION_DEFAULTS, **config}
        return self._run_dip(image, mask, **config)

    def stitch(
        self, composite: np.ndarray, gap_mask: np.ndarray, **config
    ) -> np.ndarray:
        """Fill gap in stitched composite."""
        config = {**DIP_STITCHING_DEFAULTS, **config}
        return self._run_dip(composite, gap_mask, **config)

    def enhance(
        self, image: np.ndarray, mask: np.ndarray, **config
    ) -> np.ndarray:
        """Enhance image details (full-image processing)."""
        config = {**DIP_ENHANCEMENT_DEFAULTS, **config}
        return self._run_dip(image, mask, **config)
