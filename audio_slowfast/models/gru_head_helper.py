from typing import List
from loguru import logger

import torch
import torch.nn as nn


class GRUResNetBasicHead(nn.Module):
    """
    ResNe(X)t 2D head with a GRU before.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
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
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p frequency temporal
                poolings, temporal pool kernel size, frequency pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            gru_hidden_size (int): the hidden size of the GRU. It has to match the size of the noun embeddings, which are CLIP embeddings.
            gru_num_layers (int): the number of layers of the GRU
        """
        super(GRUResNetBasicHead, self).__init__()
        assert len({len(pool_size), len(dim_in)}) == 1, "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool2d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # GRU Module
        self.gru = nn.GRU(
            input_size=sum(dim_in),  # Assuming the input size is the sum of the dimensions of the pathways
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,  # Assuming that the first dimension of the input is the batch
            bidirectional=True,  # To prevent labelling of empty frames
        )

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        assert (
            len(num_classes) == 3
        ), f"num_classes must be a list of length 3 (for (verb, noun, state)) but was {len(num_classes)}"
        self.num_classes = num_classes
        self.dim_in = dim_in

        if isinstance(self.num_classes, (list, tuple)):
            self.projection_verb = nn.Linear(sum(self.dim_in), self.num_classes[0], bias=True)
            self.projection_noun = nn.Linear(sum(self.dim_in), self.num_classes[1], bias=True)
            self.projection_state = nn.Linear(sum(dim_in), self.num_classes[2], bias=True)
        else:
            self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=3)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()

        else:
            raise NotImplementedError("{} is not supported as an activation" "function.".format(act_func))
        # State vectors use tanh activation to fall in the range [-1, 1]
        self.state_act = nn.Tanh()

    def forward(self, inputs: torch.Tensor, noun_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the GRU ResNet head. It first passes the spectrograms embeddings through the GRU to output
        a temporal sequence. The GRU is initialized with the nouns CLIP embeddings to focus attention to specific objects.
        Then the temporal sequence is passed through 3 fully connected layers, for the verb, noun and state vector respectively.

        "Projecting" here means that we are reducing the dimensionality of the input tensor to the number of classes.
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

        # GRU Module
        # Reshape noun_embeddings to be (2 * num_gpu_layers, batch_size, embedding_size)
        bidirectional_coef = 2 if self.gru.bidirectional else 1
        hidden_state_init = noun_embeddings.unsqueeze(0).repeat(bidirectional_coef * self.gru_num_layers, 1, 1)

        # Feed each spectrogram to the GRU, with the initial hidden state being the noun embeddings.
        # The GRU expects a tensor of the shape (batch, seq_len, n_features)

        # x is of shape (batch * seq_len, 1, 1, n_features_asf)
        logger.debug(f"GRU input shape: {x.shape}")

        x, _ = self.gru(x, hx=hidden_state_init)

        if isinstance(self.num_classes, (list, tuple)):
            x_v = self.projection_verb(x)
            x_n = self.projection_noun(x)
            x_s = self.projection_state(x)

            x_v = self.fc_inference(x_v, self.act)
            x_n = self.fc_inference(x_n, self.act)
            # State vectors use tanh activation to fall in the range [-1, 1]
            x_s = self.fc_inference(x_s, self.state_act)

            return (x_v, x_n, x_s)
        else:
            x = self.projection(x)
            x = self.fc_inference(x, self.act)

            return x

    def fc_inference(self, x: torch.Tensor, act: nn.Module) -> torch.Tensor:
        # Performs fully convolutional inference.
        if not self.training:
            x = act(x)
            x = x.mean([1, 2])

        x = x.view(x.shape[0], -1)
        return x
