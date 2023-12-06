import os

import h5py
from .epickitchens_record_gru import EpicKitchensAudioRecordGRU
from .epickitchens import EpicKitchens
from .build import DATASET_REGISTRY
from fvcore.common.config import CfgNode


@DATASET_REGISTRY.register()
class EpicKitchensGRU(EpicKitchens):
    def __init__(
        self,
        cfg: CfgNode,
        mode: str,
        unique_batch: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            mode=mode,
            record_type=EpicKitchensAudioRecordGRU,
            unique_batch=unique_batch,
            gru_format=True,
        )
