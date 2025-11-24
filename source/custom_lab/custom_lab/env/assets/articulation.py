from isaaclab.assets import Articulation
from custom_lab.env.assets.articulation_data import ArticulationDataWithCM
from isaaclab.utils import configclass
import torch


class ArticulationWithCM(Articulation):
    """Articulation class that includes centroidal momentum information."""

    def _initialize_impl(self):
        super()._initialize_impl()
        # Replace the base ArticulationData with ArticulationDataWithCM, preserving existing attributes
        base_data = self._data  # ArticulationData
        cm_data = ArticulationDataWithCM(self.root_physx_view, self.device)
        for key, value in base_data.__dict__.items():
            if not hasattr(cm_data, key) or getattr(cm_data, key) is None:
                setattr(cm_data, key, value)
        self._data = cm_data

    @property
    def data(self) -> ArticulationDataWithCM:
        return self._data
