from typing import Optional
from dataclasses import dataclass

from lcmr.utils.guards import typechecked
from lcmr_ext.modeler.efd_module import EfdModuleConfig


@typechecked
@dataclass
class ModelerConfig:
    num_slots: int = 1
    encoder_feature_dim: int = 2048
    prediction_head_layers: int = 3
    use_single_scale: bool = False
    use_confidence: bool = True
    efd_module_config: Optional[EfdModuleConfig] = None
