from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any, Literal, Optional

LayerType = Literal["image", "labels", "points", "shapes"]
LayerParams = Dict[str, Any]  # name, colormap, opacity, visible, etc.
Layer = Tuple[np.ndarray, LayerParams, LayerType]


class SatellitePlugin(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """The name that will be displayed in the viewer"""
        pass

    @property
    def requires_viewer(self) -> bool:
        """
        Whether this plugin needs access to the full viewer.
        Default: False (only gets the current image)
        Override to True if you need to access multiple layers.
        """
        return False

    @abstractmethod
    def run(self, image: np.ndarray, viewer=None) -> List[Layer]:
        """
        The main function that will be called when the plugin is run.

        Args:
            image: The primary input image (first layer)
            viewer: Optional napari viewer (only provided if requires_viewer=True)

        Returns:
            List of (data, params, layer_type) tuples, for the napari viewer.
            - data: np.ndarray
            - params: dict with name, colormap, opacity, etc.
            - layer_type: "image" | "labels" | ...

        Docs:
            https://napari.org/dev/howtos/layers/image.html
            https://napari.org/stable/howtos/layers/labels.html
            ...
        """
        pass
