"""Base widget class with shared UI components."""

import glob
import os

import numpy as np
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
)
from napari.qt.threading import create_worker

from ..utils import load_image_as_rgb


MODEL_OPTIONS = {
    "Deep Image Prior": {"engine": "Deep Image Prior", "use_controlnet": False},
    "Diffusion": {"engine": "Diffusion", "use_controlnet": False},
    "Diffusion + ControlNet": {"engine": "Diffusion", "use_controlnet": True},
}


class BaseComponentWidget(QWidget):
    """Base class for processing widgets with common UI patterns."""

    component_name = "Component"
    run_button_text = "Run"
    output_prefix = "[Output]"
    default_image = "train_0.tif"  # Default image to preselect

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer

    def _create_standard_ui(self):
        """Create standard UI: image source → model selector → run button."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.source_group, self.image_combo = self._create_image_source_group(layout)
        self.engine_combo = self._create_engine_selector(layout)
        self.run_button = self._create_run_button(layout)

        self._connect_standard_signals()
        self._update_image_choices(self.image_combo)

    def _connect_standard_signals(self):
        """Connect standard signals for simple widgets."""
        self.source_group.buttonClicked.connect(
            lambda: self._update_image_choices(self.image_combo)
        )
        self.run_button.clicked.connect(self._run)
        self.viewer.layers.events.inserted.connect(
            lambda _: self._update_image_choices(self.image_combo)
        )
        self.viewer.layers.events.removed.connect(
            lambda _: self._update_image_choices(self.image_combo)
        )

    def _create_image_source_group(self, layout, label="Image:"):
        """Create image source selection UI (file vs layer)."""
        group = QGroupBox("Image Selection")
        group_layout = QVBoxLayout(group)

        # Source type radio buttons
        source_layout = QHBoxLayout()
        self._source_group = QButtonGroup(self)
        self._file_radio = QRadioButton("From File")
        self._layer_radio = QRadioButton("From Layer")
        self._file_radio.setChecked(True)
        self._source_group.addButton(self._file_radio)
        self._source_group.addButton(self._layer_radio)
        source_layout.addWidget(self._file_radio)
        source_layout.addWidget(self._layer_radio)
        source_layout.addStretch()
        group_layout.addLayout(source_layout)

        # Image selection dropdown
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel(label))
        combo = QComboBox()
        selection_layout.addWidget(combo, 1)
        group_layout.addLayout(selection_layout)
        layout.addWidget(group)

        return self._source_group, combo

    def _create_engine_selector(self, layout):
        """Create engine selection dropdown."""
        group = QGroupBox("Model Selection")
        group_layout = QVBoxLayout(group)

        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Model:"))
        combo = QComboBox()
        combo.addItems(list(MODEL_OPTIONS.keys()))
        selection_layout.addWidget(combo, 1)

        group_layout.addLayout(selection_layout)
        layout.addWidget(group)
        return combo

    def _create_run_button(self, layout, text=None):
        """Create the main action button."""
        button = QPushButton(text or self.run_button_text)
        button.setMinimumHeight(40)
        layout.addWidget(button)
        return button

    def _update_image_choices(self, combo, default=None):
        """Refresh image dropdown based on selected source."""
        if self._file_radio.isChecked():
            choices = [os.path.basename(p) for p in sorted(glob.glob("data/marcin/*.tif"))]
        else:
            choices = [
                layer.name for layer in self.viewer.layers
                if hasattr(layer, "data") and layer.data.ndim >= 2
            ]

        current = combo.currentText()
        combo.clear()
        # Add placeholder as first item
        combo.addItem("-- Select Image --")
        if choices:
            combo.addItems(choices)
        # Restore previous selection, or use default
        if current in choices:
            combo.setCurrentText(current)
        elif (default or self.default_image) in choices:
            combo.setCurrentText(default or self.default_image)

    def _get_model_config(self):
        """Get model configuration from selected option."""
        name = self.engine_combo.currentText()
        return MODEL_OPTIONS.get(name, MODEL_OPTIONS["Deep Image Prior"])

    def _get_engine_name(self):
        return self._get_model_config()["engine"]

    def _get_use_controlnet(self):
        return self._get_model_config()["use_controlnet"]

    def _load_image(self):
        """Load image from file or layer based on current selection."""
        selection = self.image_combo.currentText()
        if not selection or selection.startswith("-- Select"):
            return None

        if self._file_radio.isChecked():
            path = f"data/marcin/{selection}"
            return load_image_as_rgb(path) if os.path.exists(path) else None

        for layer in self.viewer.layers:
            if layer.name == selection:
                data = layer.data.astype(np.float32)
                if data.max() > 1:
                    data = data / 255.0
                return data
        return None

    def _run_in_worker(self, function, button=None):
        """Run function in background thread."""
        button = button or self.run_button
        original_text = button.text()
        button.setText("Processing...")
        button.setEnabled(False)

        def on_done(result):
            button.setText(original_text)
            button.setEnabled(True)
            if result:
                self._add_output_layers(*result)
         # come onnnnn
        def on_error(error):
            button.setText(original_text)
            button.setEnabled(True)
            print(f"Error: {error}")

        worker = create_worker(function)
        worker.returned.connect(on_done)
        worker.errored.connect(on_error)
        worker.start()

    def _add_output_layers(self, input_image, output_image):
        """Add input and output images as viewer layers."""
        self.viewer.add_image(input_image, name=f"{self.output_prefix} Input", rgb=True)
        self.viewer.add_image(output_image, name=f"{self.output_prefix} Output", rgb=True)

    def _run(self):
        """Override in subclass."""
        raise NotImplementedError
