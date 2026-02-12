"""Enhancement widget for napari viewer."""

from .base import BaseComponentWidget
from ..components.enhancement import EnhancementProcessor

ENHANCEMENT_CONFIGS = {
    "Deep Image Prior": {"hf_blend": 0.5, "num_iter": 800, "lr": 0.005},
    "Diffusion": {"hf_blend": 0.9, "num_steps": 25, "strength": 0.4, "use_controlnet": False},
    "Diffusion + ControlNet": {"hf_blend": 0.9, "num_steps": 25, "strength": 0.4,
                               "use_controlnet": True, "cn_scale": 1.2, "control_guidance_end": 0.9},
}


class EnhancementWidget(BaseComponentWidget):
    component_name = "Detail Enhancement"
    run_button_text = "Run Enhancement"
    output_prefix = "[Enhancement]"

    def __init__(self, viewer, parent=None):
        super().__init__(viewer, parent)
        self._create_standard_ui()

    def _run(self):
        image = self._load_image()
        if image is None:
            return

        model_name = self.engine_combo.currentText()
        config = ENHANCEMENT_CONFIGS.get(model_name, {}).copy()
        hf_blend = config.pop("hf_blend", 0.5)

        def process():
            processor = EnhancementProcessor(self._get_engine_name())
            return processor.process(image, hf_blend=hf_blend, **config)

        self._run_in_worker(process)


def create_enhancement_widget(viewer):
    return EnhancementWidget(viewer)
