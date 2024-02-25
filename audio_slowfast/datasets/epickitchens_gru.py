from fvcore.common.config import CfgNode

from .build import DATASET_REGISTRY
from .epickitchens import EpicKitchens
from .epickitchens_record_gru import EpicKitchensAudioRecordGRU


@DATASET_REGISTRY.register()
class EpicKitchensGRU(EpicKitchens):
    def __init__(
        self,
        cfg: CfgNode,
        mode: str,
    ):
        super().__init__(
            cfg=cfg,
            mode=mode,
            record_type=EpicKitchensAudioRecordGRU,
            gru_format=True,
        )
