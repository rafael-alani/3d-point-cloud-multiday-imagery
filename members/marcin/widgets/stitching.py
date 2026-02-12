"""Stitching widget for napari viewer."""

from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QComboBox

from .base import BaseComponentWidget
from ..components.stitching import StitchingProcessor

OVERLAP = 128
BLEND_WIDTH = 5


class StitchingWidget(BaseComponentWidget):
    component_name = "Image Stitching"
    run_button_text = "Run Stitching"
    output_prefix = "[Stitching]"
    default_left = "train_0.tif"
    default_right = "train_147.tif"

    def __init__(self, viewer, parent=None):
        super().__init__(viewer, parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Source type (reuse base class pattern)
        self.source_group, self.image_combo = self._create_image_source_group(layout)

        # Replace single combo with left/right combos
        self.image_combo.deleteLater()

        # Get the group widget and its layout
        image_group = layout.itemAt(0).widget()
        image_layout = image_group.layout()

        # Remove the "Image:" row that was created
        item = image_layout.takeAt(1)
        if item and item.layout():
            while item.layout().count():
                child = item.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        # Add left/right combos
        self.left_combo = QComboBox()
        self.right_combo = QComboBox()

        left_row = QHBoxLayout()
        left_row.addWidget(QLabel("Left:"))
        left_row.addWidget(self.left_combo, 1)
        image_layout.addLayout(left_row)

        right_row = QHBoxLayout()
        right_row.addWidget(QLabel("Right:"))
        right_row.addWidget(self.right_combo, 1)
        image_layout.addLayout(right_row)

        # Model selection
        self.engine_combo = self._create_engine_selector(layout)

        # Run button
        self.run_button = self._create_run_button(layout)

        # Signals
        self.source_group.buttonClicked.connect(self._update_combos)
        self.run_button.clicked.connect(self._run)
        self.viewer.layers.events.inserted.connect(lambda _: self._update_combos())
        self.viewer.layers.events.removed.connect(lambda _: self._update_combos())

        self._update_combos()

    def _update_combos(self):
        self._update_image_choices(self.left_combo, default=self.default_left)
        self._update_image_choices(self.right_combo, default=self.default_right)

    def _load_images(self):
        images = []
        for combo in [self.left_combo, self.right_combo]:
            self.image_combo = combo  # temp for _load_image
            img = self._load_image()
            if img is None:
                return None
            images.append(img)
        return images

    def _run(self):
        images = self._load_images()
        if not images:
            return

        def process():
            processor = StitchingProcessor(self._get_engine_name())
            return processor.process(
                images,
                overlap=OVERLAP, blend_width=BLEND_WIDTH,
                use_controlnet=self._get_use_controlnet(),
            )

        self._run_in_worker(process)


def create_stitching_widget(viewer):
    return StitchingWidget(viewer)
