from typing import List
from loguru import logger

import torch
import torch.nn as nn

from audio_slowfast.utils import discretize

from .models.head_helper import ResNetBasicHead


class GRUResNetBasicHead(ResNetBasicHead):
    """
    ResNet basic head for audio classification.

    Overrides the original head defined in
    auditory_slow_fast/audio_slowfast/models/head_helper.py
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        gru_hidden_size=512,
        gru_num_layers=2,
    ) -> None:
        super(GRUResNetBasicHead, self).__init__(
            dim_in=dim_in,
            num_classes=num_classes,
            pool_size=pool_size,
            dropout_rate=dropout_rate,
            act_func=act_func,
        )
        # GRU Module
        self.gru = nn.GRU(
            input_size=sum(dim_in),  # Assuming the input size is the sum of the dimensions of the pathways
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,  # Assuming that the first dimension of the input is the batch
            bidirectional=True,  # To fight hallucinations
        )

        if len(num_classes) == 4:
            logger.info("Using 4 classification heads")

            self.projection_prec = nn.Linear(sum(dim_in), num_classes[2], bias=True)
            self.projection_postc = nn.Linear(sum(dim_in), num_classes[3], bias=True)

    def forward(self, inputs: torch.Tensor, noun_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the ResNet basic head.

        Parameters
        ----------
        `inputs` : `list` of `torch.Tensor`
            List of input tensors.

        `noun_embeddings` : `torch.Tensor`
            Noun embeddings.

        Returns
        -------
        `torch.Tensor`
            Output tensor.
        """
        assert len(inputs) == self.num_pathways, "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H) -> (N, T, H, C).
        x = x.permute((0, 2, 3, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        # Pass through GRU
        x, _ = self.gru(x)

        return x

    def _proj(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """
        Projection function

        Parameters
        ----------
        `x` : `torch.Tensor`
            Input tensor.
        `proj` : `nn.Module`
            Projection function

        Returns
        -------
        `torch.Tensor`
            Projected view of input tensor.
        """
        x_v = proj(x)

        # Performs fully convolutional inference.
        x_v: torch.Tensor = self.act(x_v)

        if not self.training:
            x_v = x_v.mean([1, 2])

        return x_v.view(x_v.shape[0], -1)

    def _proj_discrete(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        x_v = proj(x)
        # Apply tanh activation
        x_v = torch.tanh(x_v)
        # Discretize the output
        x_v = discretize(x_v)
        return x_v.view(x_v.shape[0], -1)

    def project_pre_post_conditions(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        """
        Project the input tensor to pre- and post-conditions respective classes.

        Parameters
        ----------
        `x` : `torch.Tensor`
            Input tensor.

        Returns
        -------
        `torch.Tensor | List[torch.Tensor]`
            Projected view of input tensor.
        """

        if isinstance(self.num_classes, (list, tuple)):
            return (
                self._proj(x, self.projection_verb),
                self._proj(x, self.projection_noun),
                self._proj(x, self.projection_prec),
                self._proj(x, self.projection_postc),
                # self._proj_discrete(x, self.projection_prec),
                # self._proj_discrete(x, self.projection_postc),
            )

        return self._proj(x, self.projection)
