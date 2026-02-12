"""Restoration widget for napari viewer."""

import os
import numpy as np

from .base import BaseComponentWidget
from ..components.restoration import ImageRestorationProcessor
from ..utils import load_image_with_nans

MARGIN = 10


class ImageRestorationWidget(BaseComponentWidget):
    component_name = "Image Restoration"
    run_button_text = "Run Restoration"
    output_prefix = "[Restoration]"

    def __init__(self, viewer, parent=None):
        super().__init__(viewer, parent)
        self._create_standard_ui()

    def _load_image_with_nans(self):
        """Load image and NaN mask from current selection."""
        selection = self.image_combo.currentText()
        if not selection or selection.startswith("-- Select"):
            return None

        if self._file_radio.isChecked():
            path = f"data/marcin/{selection}"
            return load_image_with_nans(path) if os.path.exists(path) else None

        # Load from layer
        for layer in self.viewer.layers:
            if layer.name == selection:
                data = layer.data.copy()
                nan_mask = np.any(np.isnan(data), axis=2) if data.ndim == 3 else np.isnan(data)
                data = np.nan_to_num(data, nan=0.0)
                if data.max() > 1:
                    data = data / 255.0
                return data.astype(np.float32), nan_mask
        return None

    def _run(self):
        result = self._load_image_with_nans()
        if not result:
            return

        image, nan_mask = result
        if not nan_mask.any():
            return

        def process():
            processor = ImageRestorationProcessor(self._get_engine_name())
            return processor.process(
                image, nan_mask,
                margin=MARGIN,
                use_controlnet=self._get_use_controlnet(),
            )

        self._run_in_worker(process)


def create_restoration_widget(viewer):
    return ImageRestorationWidget(viewer)
