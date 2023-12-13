import torch
import torch.utils.data
from . import utils as utils
from .epickitchens_record_pddl import EpicKitchensAudioRecordWithPDDL
from .epickitchens import EpicKitchens
from .build import DATASET_REGISTRY
from fvcore.common.config import CfgNode


@DATASET_REGISTRY.register()
class EpicKitchensWithPDDL(EpicKitchens):
    def __init__(
        self,
        cfg: CfgNode,
        mode: str,
    ):
        super().__init__(
            cfg=cfg,
            mode=mode,
            record_type=EpicKitchensAudioRecordWithPDDL,
            gru_format=False,
        )
