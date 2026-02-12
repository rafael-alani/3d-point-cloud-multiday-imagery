import glob
import os
from pathlib import Path

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QGroupBox,
)
from napari.qt.threading import create_worker
from . import constants as C

class DisparityWidget(QWidget):
    """Incorporate the ability to set the debugging parameters and not
    have to touch the code."""

    def __init__(self, viewer, plugin, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.plugin = plugin
        self._create_ui()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # KML Selection
        kml_group = QGroupBox("KML File Selection")
        kml_layout = QVBoxLayout(kml_group)
        self.kml_combo = QComboBox()
        self._update_kml_choices()
        kml_layout.addWidget(self.kml_combo)
        layout.addWidget(kml_group)

        # Debug Options
        debug_group = QGroupBox("Debug Options")
        debug_layout = QVBoxLayout(debug_group)
        
        self.chk_debug_mode = QCheckBox("Debug Mode (Show intermediary images)")
        self.chk_debug_mode.setChecked(C.IS_DEBUG_MODE)
        debug_layout.addWidget(self.chk_debug_mode)

        self.chk_debug_pair = QCheckBox("Debug Pair (Used during development)")
        self.chk_debug_pair.setChecked(C.IS_DEBUG_PAIR)
        debug_layout.addWidget(self.chk_debug_pair)

        self.chk_one_random = QCheckBox("Process One Random Pair \n \
        (Disable to compute N point clouds)")
        self.chk_one_random.setChecked(C.IS_ONE_RANDOM_PAIR)
        debug_layout.addWidget(self.chk_one_random)
        
        layout.addWidget(debug_group)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("Number of Point Clouds:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 100)
        self.spin_n.setValue(C.N)
        param_layout.addWidget(self.spin_n)
        layout.addWidget(param_group)

        # Run Button
        self.run_button = QPushButton(f"Run {self.plugin.name}")
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self._run)
        layout.addWidget(self.run_button)

        layout.addStretch()

    def _update_kml_choices(self):
        """Find KLM files"""
        patterns = ["data/**/*.kml", "**/*.kml"]
        files = []
        for pattern in patterns:
            files.extend(sorted(glob.glob(pattern, recursive=True)))
        files = list(dict.fromkeys(files))
        
        self.kml_combo.clear()
        self.kml_combo.addItem("-- Select KML File --")
        if files:
            self.kml_combo.addItems(files)
        
        # Default selection if available
        default_kml = "data/rafael/Explorer.kml"
        for f in files:
            if f.endswith("Explorer.kml"):
                self.kml_combo.setCurrentText(f)
                break

    def _run(self):
        kml_selection = self.kml_combo.currentText()
        if not kml_selection or kml_selection.startswith("-- Select"):
            print("Please select a KML file.")
            return
            
        kml_path = Path(kml_selection)
        
        # Collect params
        is_debug_mode = self.chk_debug_mode.isChecked()
        is_debug_pair = self.chk_debug_pair.isChecked()
        is_one_random = self.chk_one_random.isChecked()
        n = self.spin_n.value()

        original_text = self.run_button.text()
        self.run_button.setText("Processing...")
        self.run_button.setEnabled(False)

        def process():
            return self.plugin.run(
                kml_path, 
                is_debug_mode=is_debug_mode,
                is_debug_pair=is_debug_pair,
                is_one_random_pair=is_one_random,
                n=n
            )

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
                print(f"Error adding layers: {e}")
                import traceback
                traceback.print_exc()

        def on_error(error):
            self.run_button.setText(original_text)
            self.run_button.setEnabled(True)
            print(f"Error running plugin: {error}")

        worker = create_worker(process)
        worker.returned.connect(on_done)
        worker.errored.connect(on_error)
        worker.start()

def create_disparity_widget(viewer, plugin):
    return DisparityWidget(viewer, plugin)
