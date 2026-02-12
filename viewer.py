"""Main entry point for satellite image restoration viewer."""

import os
import glob

import napari
import tifffile
import numpy as np
from magicgui import magicgui
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QGroupBox,
    QTabWidget,
    QSizePolicy,
)
from napari.qt.threading import create_worker

# IMPORTANT: Import GDAL-dependent plugins FIRST before other imports
# that may load conflicting native libraries (libtiff)
RAFAEL_PLUGINS = []

try:
    from members.rafael.disparity import HeightMapExtractor, DisparityWidget
    RAFAEL_PLUGINS.append(HeightMapExtractor())
except Exception as e:
    print(f"HeightMapExtractor not available: {e}")

try:
    from members.rafael.saliency_object_annotation import SaliencyDetector
    RAFAEL_PLUGINS.append(SaliencyDetector())
except Exception as e:
    print(f"SaliencyDetector not available: {e}")

# Stan's plugins
STAN_PLUGINS = []

try:
    from members.stan import StanInpainter, StanSuperRes
    STAN_PLUGINS.append(StanInpainter())
    STAN_PLUGINS.append(StanSuperRes())
except Exception as e:
    print(f"Stan plugins not available: {e}")

# Jasraj's plugins
JASRAJ_PLUGINS = []

try:
    from members.jasraj import (
        RestorationPlugin,
        ImageStitchingPlugin,
        LandUseClassificationPlugin,
        ObjectAnnotationPlugin,
    )
    JASRAJ_PLUGINS.append(RestorationPlugin())
    JASRAJ_PLUGINS.append(ImageStitchingPlugin())
    JASRAJ_PLUGINS.append(LandUseClassificationPlugin())
    JASRAJ_PLUGINS.append(ObjectAnnotationPlugin())
except Exception as e:
    print(f"Jasraj plugins not available: {e}")

# Marcin's widgets (import after GDAL to avoid libtiff conflicts)
from members.marcin.widgets import (
    create_restoration_widget,
    create_stitching_widget,
    create_enhancement_widget,
)


def normalize_band(band):
    """Normalize band to 0-1 using 2-98 percentile stretch."""

    band = np.nan_to_num(band, nan=0.0)

    valid_pixels = band > 0
    if not valid_pixels.any():
        return np.zeros_like(band)

    low, high = np.percentile(band[valid_pixels], [2, 98])

    if high == low:
        return np.zeros_like(band)

    normalized = (band - low) / (high - low)
    return np.clip(normalized, 0, 1)


class PluginWidget(QWidget):
    """Widget wrapper for SatellitePlugin with file/layer selection."""

    def __init__(self, viewer, plugin, data_path="data", default_image=None, default_image2=None, default_extras=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.plugin = plugin
        self.data_path = data_path  # Member-specific data folder
        self.default_image = default_image  # Default image to select
        self.default_image2 = default_image2  # Default second image
        self.default_extras = default_extras or {}  # Default extra files (e.g., kml)
        self._extra_file_combos = {}
        # Check if plugin requires image selection
        self._requires_image = getattr(plugin, 'requires_image', True)
        # Check if plugin needs a second image
        self._requires_image2 = self._check_needs_image2()
        self._create_ui()

    def _check_needs_image2(self):
        """Check if plugin.run() accepts an image2 parameter."""
        import inspect
        sig = inspect.signature(self.plugin.run)
        return 'image2' in sig.parameters

    def _get_extra_params(self):
        """Get extra parameters from plugin.run() signature (beyond 'image' and 'image2')."""
        import inspect
        sig = inspect.signature(self.plugin.run)
        extras = []
        for name, param in sig.parameters.items():
            if name in ('self', 'image', 'image2'):
                continue
            extras.append((name, param.annotation))
        return extras

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Image source selection (only if plugin requires it)
        if self._requires_image:
            source_group = QGroupBox("Image Selection")
            source_layout = QVBoxLayout(source_group)

            # Radio buttons for file vs layer
            radio_layout = QHBoxLayout()
            self._source_group = QButtonGroup(self)
            self._file_radio = QRadioButton("From File")
            self._layer_radio = QRadioButton("From Layer")
            self._file_radio.setChecked(True)
            self._source_group.addButton(self._file_radio)
            self._source_group.addButton(self._layer_radio)
            radio_layout.addWidget(self._file_radio)
            radio_layout.addWidget(self._layer_radio)
            radio_layout.addStretch()
            source_layout.addLayout(radio_layout)

            # Image dropdown (label changes if we have image2)
            selection_layout = QHBoxLayout()
            label = "Image 1:" if self._requires_image2 else "Image:"
            selection_layout.addWidget(QLabel(label))
            self.image_combo = QComboBox()
            selection_layout.addWidget(self.image_combo, 1)
            source_layout.addLayout(selection_layout)

            # Second image dropdown if needed
            self.image2_combo = None
            if self._requires_image2:
                selection2_layout = QHBoxLayout()
                selection2_layout.addWidget(QLabel("Image 2:"))
                self.image2_combo = QComboBox()
                selection2_layout.addWidget(self.image2_combo, 1)
                source_layout.addLayout(selection2_layout)

            layout.addWidget(source_group)
        else:
            self.image_combo = None
            self.image2_combo = None
            self._file_radio = None
            self._layer_radio = None
            self._source_group = None

        # Extra file parameters (e.g., kml_path)
        extra_params = self._get_extra_params()
        if extra_params:
            extras_group = QGroupBox("Additional Files")
            extras_layout = QVBoxLayout(extras_group)
            for param_name, param_type in extra_params:
                row = QHBoxLayout()
                label = param_name.replace('_', ' ').title() + ":"
                row.addWidget(QLabel(label))
                combo = QComboBox()
                row.addWidget(combo, 1)
                extras_layout.addLayout(row)
                self._extra_file_combos[param_name] = combo
            layout.addWidget(extras_group)

        # Run button
        self.run_button = QPushButton(f"Run {self.plugin.name}")
        self.run_button.setMinimumHeight(40)
        layout.addWidget(self.run_button)

        # Connect signals
        self.run_button.clicked.connect(self._run)
        if self._requires_image:
            self._source_group.buttonClicked.connect(self._update_image_choices)
            self.viewer.layers.events.inserted.connect(lambda _: self._update_image_choices())
            self.viewer.layers.events.removed.connect(lambda _: self._update_image_choices())
            self._update_image_choices()
        else:
            self._update_extra_file_choices()

    def _update_extra_file_choices(self):
        """Update extra file combos (e.g., kml files)."""
        for param_name, combo in self._extra_file_combos.items():
            # Determine file pattern based on parameter name
            if 'kml' in param_name.lower():
                patterns = ["data/**/*.kml", "**/*.kml"]
            else:
                patterns = ["data/**/*"]

            files = []
            for pattern in patterns:
                files.extend(sorted(glob.glob(pattern, recursive=True)))
            files = list(dict.fromkeys(files))  # Remove duplicates

            current = combo.currentText()
            combo.clear()
            # Add placeholder
            combo.addItem("-- Select File --")
            if files:
                combo.addItems(files)
            # Restore previous selection or use default
            if current in files:
                combo.setCurrentText(current)
            elif param_name in self.default_extras:
                default = self.default_extras[param_name]
                for f in files:
                    if f == default or f.endswith(default):
                        combo.setCurrentText(f)
                        break

    def _update_image_choices(self):
        """Refresh image dropdown based on selected source."""
        if self._file_radio.isChecked():
            # Look in member-specific data directory
            choices = []
            for ext in ["*.tif", "*.png", "*.jpg", "*.webp"]:
                pattern = f"{self.data_path}/{ext}"
                choices.extend([p for p in sorted(glob.glob(pattern))])
                # Also check subdirectories
                pattern = f"{self.data_path}/**/{ext}"
                choices.extend([p for p in sorted(glob.glob(pattern, recursive=True))])
            choices = list(dict.fromkeys(choices))  # Remove duplicates, keep order
        else:
            choices = [
                layer.name for layer in self.viewer.layers
                if hasattr(layer, "data") and isinstance(layer.data, np.ndarray) and layer.data.ndim >= 2
            ]

        # Update first image combo
        current = self.image_combo.currentText()
        self.image_combo.clear()
        self.image_combo.addItem("-- Select Image --")
        if choices:
            self.image_combo.addItems(choices)
        # Restore previous selection or use default
        if current in choices:
            self.image_combo.setCurrentText(current)
        elif self.default_image:
            # Find matching choice (check both full path and filename)
            for choice in choices:
                if choice == self.default_image or choice.endswith(self.default_image):
                    self.image_combo.setCurrentText(choice)
                    break

        # Update second image combo if present
        if self.image2_combo is not None:
            current2 = self.image2_combo.currentText()
            self.image2_combo.clear()
            self.image2_combo.addItem("-- Select Image --")
            if choices:
                self.image2_combo.addItems(choices)
            # Restore previous selection or use default
            if current2 in choices:
                self.image2_combo.setCurrentText(current2)
            elif self.default_image2:
                for choice in choices:
                    if choice == self.default_image2 or choice.endswith(self.default_image2):
                        self.image2_combo.setCurrentText(choice)
                        break

        # Also update extra file combos
        self._update_extra_file_choices()

    def _load_image_from_combo(self, combo):
        """Load image from file or layer using specified combo."""
        import cv2
        selection = combo.currentText()
        if not selection or selection.startswith("-- Select"):
            return None

        if self._file_radio.isChecked():
            # Selection is now a full path
            if os.path.exists(selection):
                if selection.endswith('.tif') or selection.endswith('.tiff'):
                    data = tifffile.imread(selection)
                else:
                    # Use cv2 for png/jpg
                    data = cv2.imread(selection)
                    if data is not None:
                        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                if data is not None:
                    return data.astype(np.float32)
            return None

        # Load from layer
        for layer in self.viewer.layers:
            if layer.name == selection:
                data = layer.data.copy()
                return data.astype(np.float32)
        return None

    def _load_image(self):
        """Load image from file or layer."""
        if not self._requires_image:
            return None
        return self._load_image_from_combo(self.image_combo)

    def _load_image2(self):
        """Load second image from file or layer."""
        if self.image2_combo is None:
            return None
        return self._load_image_from_combo(self.image2_combo)

    def _run(self):
        """Run the plugin."""
        from pathlib import Path

        image = self._load_image()
        if self._requires_image and image is None:
            print("No image selected")
            return

        # Load second image if needed
        image2 = None
        if self._requires_image2:
            image2 = self._load_image2()
            if image2 is None:
                print("No second image selected")
                return

        # Gather extra parameters
        extra_kwargs = {}
        for param_name, combo in self._extra_file_combos.items():
            file_path = combo.currentText()
            if file_path and not file_path.startswith("-- Select"):
                extra_kwargs[param_name] = Path(file_path)
            else:
                print(f"No file selected for {param_name}")
                return

        # Add image2 to kwargs if needed
        if self._requires_image2 and image2 is not None:
            extra_kwargs['image2'] = image2

        original_text = self.run_button.text()
        self.run_button.setText("Processing...")
        self.run_button.setEnabled(False)

        def process():
            if self._requires_image:
                return self.plugin.run(image, **extra_kwargs)
            else:
                return self.plugin.run(**extra_kwargs)

        def on_done(layers_to_add):
            self.run_button.setText(original_text)
            self.run_button.setEnabled(True)
            if not layers_to_add:
                return
            try:
                for data, params, layer_type in layers_to_add:
                    adder = getattr(self.viewer, f"add_{layer_type}")
                    adder(data, **params)
            except Exception as e:
                print(f"Error adding layers from {self.plugin.name}: {e}")
                import traceback
                traceback.print_exc()

        def on_error(error):
            self.run_button.setText(original_text)
            self.run_button.setEnabled(True)
            print(f"Error running {self.plugin.name}: {error}")

        worker = create_worker(process)
        worker.returned.connect(on_done)
        worker.errored.connect(on_error)
        worker.start()


def create_plugin_widget(viewer, plugin, data_path="data", default_image=None, default_image2=None, default_extras=None):
    """Create a widget for a SatellitePlugin."""
    return PluginWidget(viewer, plugin, data_path=data_path, default_image=default_image, default_image2=default_image2, default_extras=default_extras)


def main():
    viewer = napari.Viewer(title="Satellite Image Restoration")

    # Find available images (from marcin's folder for the main loader)
    image_paths = sorted(glob.glob("data/marcin/*.tif"))
    images = {os.path.basename(path): path for path in image_paths}

    @magicgui(
        call_button="Load",
        filename={"choices": list(images.keys()) or ["No images"]},
    )
    def load_image(filename: str):
        if filename == "No images" or filename not in images:
            return

        data = tifffile.imread(images[filename])

        # Clear existing layers
        viewer.layers.clear()

        # Extract and normalize RGB (Sentinel-2 bands 4, 3, 2)
        red = normalize_band(data[:, :, 3])
        green = normalize_band(data[:, :, 2])
        blue = normalize_band(data[:, :, 1])
        rgb = np.stack([red, green, blue], axis=-1)

        viewer.add_image(rgb, name="RGB", rgb=True)

    # Add load widget
    viewer.window.add_dock_widget(load_image, area="left", name="Dataset")

    # Create tabs grouped by member (alphabetical order)
    # Use Maximum size policy for widgets so they don't expand
    MAX_PANEL_WIDTH = 350

    # Jasraj's tab - each plugin has component-specific test data
    if JASRAJ_PLUGINS:
        jasraj_tabs = QTabWidget()
        jasraj_tabs.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        jasraj_tabs.setMaximumWidth(MAX_PANEL_WIDTH)
        jasraj_defaults = {
            "Image Restoration (Denoise + Dehaze)": {"default_image": "restoration/1.png"},
            "Image Stitching (SIFT + Seam Cut)": {"default_image": "image_stitching/Input1/11.png", "default_image2": "image_stitching/Input1/12.png"},
            "Land Use Classification (OBIA)": {"default_image": "land_use_classification/1.png"},
            "Object Detection (YOLO + SAHI)": {"default_image": "object_annotation/1.png"},
        }
        for plugin in JASRAJ_PLUGINS:
            defaults = jasraj_defaults.get(plugin.name, {})
            jasraj_tabs.addTab(create_plugin_widget(
                viewer, plugin, data_path="data/jasraj",
                default_image=defaults.get("default_image"),
                default_image2=defaults.get("default_image2")
            ), plugin.name)
        viewer.window.add_dock_widget(jasraj_tabs, area="right", name="Jasraj")

    # Marcin's tab
    marcin_tabs = QTabWidget()
    marcin_tabs.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
    marcin_tabs.setMaximumWidth(MAX_PANEL_WIDTH)
    marcin_tabs.addTab(create_restoration_widget(viewer), "Restoration")
    marcin_tabs.addTab(create_stitching_widget(viewer), "Stitching")
    marcin_tabs.addTab(create_enhancement_widget(viewer), "Enhancement")
    viewer.window.add_dock_widget(marcin_tabs, area="right", name="Marcin")

    # Rafael's tab
    if RAFAEL_PLUGINS:
        rafael_tabs = QTabWidget()
        rafael_tabs.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        rafael_tabs.setMaximumWidth(MAX_PANEL_WIDTH)
        rafael_defaults = {
            "Height Map": {"default_extras": {"kml_path": "data/rafael/WV3/Explorer.kml"}},
            "Saliency": {"default_image": "Saliency Object Detection/low_res_destroyer_satelite.webp"},
        }
        for plugin in RAFAEL_PLUGINS:
            defaults = rafael_defaults.get(plugin.name, {})
            if "3D Point Cloud" in plugin.name:
                 rafael_tabs.addTab(DisparityWidget(viewer, plugin), plugin.name)
            else:
                rafael_tabs.addTab(create_plugin_widget(
                    viewer, plugin, data_path="data/rafael",
                    default_image=defaults.get("default_image"),
                    default_extras=defaults.get("default_extras", {})
                ), plugin.name)
        viewer.window.add_dock_widget(rafael_tabs, area="right", name="Rafael")

    # Stan's tab - defaults to train_20.png (used in notebook demos)
    if STAN_PLUGINS:
        stan_tabs = QTabWidget()
        stan_tabs.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        stan_tabs.setMaximumWidth(MAX_PANEL_WIDTH)
        for plugin in STAN_PLUGINS:
            stan_tabs.addTab(create_plugin_widget(viewer, plugin, data_path="data/stan", default_image="train_20.png"), plugin.name)
        viewer.window.add_dock_widget(stan_tabs, area="right", name="Stan")

    napari.run()


if __name__ == "__main__":
    main()
