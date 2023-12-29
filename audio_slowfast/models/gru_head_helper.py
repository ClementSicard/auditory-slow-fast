from typing import List, Optional

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
        only_action_recognition: bool = False,
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
        self.only_action_recognition = only_action_recognition

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

        # Project back the GRU output to the number features of the input
        self.projection_to_dim_in = nn.Linear(gru_hidden_size * 2, sum(dim_in), bias=True)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        assert len(num_classes) == (
            3 if not self.only_action_recognition else 2
        ), f"num_classes must be a list of length {(3 if not self.only_action_recognition else 2)} but was {len(num_classes)}: {num_classes}"

        self.num_classes = num_classes
        self.dim_in = dim_in

        # At this stage, a batch has shape (B * N_s, 1, )

        if isinstance(self.num_classes, (list, tuple)):
            F = sum(self.dim_in)

            if not self.only_action_recognition:
                V, N, P = self.num_classes
                self.projection_min_1 = nn.Linear(F, P, bias=True)
                self.projection_0 = nn.Linear(F, P, bias=True)
                self.projection_1 = nn.Linear(F, P, bias=True)

                self.state_act = nn.Softmax(dim=2)  # Softmax on [B, N, 3, P]
            else:
                V, N = self.num_classes

            self.projection_verb = nn.Linear(F, V, bias=True)
            self.projection_noun = nn.Linear(F, N, bias=True)

        else:
            self.projection = nn.Linear(F, num_classes, bias=True)
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            # Dimension 3 is the size of the embedding space before the projection.
            # A batch has the shape (B * N_s, 1, 1, 2304) -> 2304
            self.act = nn.Softmax(dim=3)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()

        else:
            raise NotImplementedError(f"{act_func} is not supported as an activation function")

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: List[int],
        initial_batch_shape: torch.Size,
        noun_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        # (B*N, F, 1, 1) -> (B*N, 1, 1, F)
        x = x.permute((0, 2, 3, 1))

        """
        Starting from here, x has shape (B * N_s, 1, 1, n_features_asf)
        """

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self._gru(
            x=x,
            noun_embeddings=noun_embeddings,
            initial_batch_shape=initial_batch_shape,
            lengths=lengths,
        )

        if isinstance(self.num_classes, (list, tuple)):
            # Recover B, N
            B, N = initial_batch_shape

            if not self.only_action_recognition:
                N_v, N_n, N_p = self.num_classes
            else:
                N_v, N_n = self.num_classes

            x_v = self.projection_verb(x)  # (B*N, 1, 1, F) -> (B*N, 1, 1, N_v)
            x_v = self.fc_inference(x_v, self.act)
            x_v = x_v.view(B, N, N_v)  # (B*N, 1, 1, N_v) -> (B, N, N_v)

            x_n = self.projection_noun(x)  # (B*N, 1, 1, F) -> (B*N, 1, 1, N_n)
            x_n = self.fc_inference(x_n, self.act)
            x_n = x_n.view(B, N, N_n)  # (B*N, 1, 1, N_n) -> (B, N, N_n)

            x_n_mean = torch.zeros(B, N_n).to(x_n.device)
            x_v_mean = torch.zeros(B, N_v).to(x_v.device)
            for i, length in enumerate(lengths):
                # First index is slected, hence dim=0 for 1st dimension
                x_n_mean[i] = x_n[i, :length, :].mean(dim=0)
                x_v_mean[i] = x_v[i, :length, :].mean(dim=0)

            x_n = x_n_mean
            x_v = x_v_mean

            assert x_n.shape == (B, N_n), f"x_n.shape must be {(B, N_n)} but was {x_n.shape}"
            assert x_v.shape == (B, N_v), f"x_v.shape must be {(B, N_v)} but was {x_v.shape}"

            if not self.only_action_recognition:
                x_s = self.project_state(x)  # (B*N, 1, 1, F) -> (B*N, 1, 3, P)
                x_s = self.fc_inference_state(x_s, self.state_act)  # (B*N, 1, 3, P) -> (B*N, 3, P)
                x_s = x_s.view(B, N, N_p, 3)  # (B*N, 3, P) -> (B, N, P, 3)

                return (x_v, x_n, x_s)

            return (x_v, x_n)

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

    def fc_inference_state(self, x: torch.Tensor, act: nn.Module) -> torch.Tensor:
        # Performs fully convolutional inference.
        # At the beginning, x has shape (B*N, 1, 3, P)
        if not self.training:
            x = act(x)  # (B*N, 1, 3, P) -> (B*N, 1, 3, P)

        x = x.mean(1)  # (B*N, 1, 3, P) -> (B*N, 3, P)

        return x

    def _gru(
        self,
        x: torch.Tensor,
        noun_embeddings: torch.Tensor,
        initial_batch_shape: torch.Size,
        lengths: List[int],
    ) -> torch.Tensor:
        """
        From Table 1 of the paper (https://arxiv.org/pdf/2103.03516.pdf), n_features_asf = 2304
        (2048 (Slow) + 256 (Fast)) just before pooling, concatenation and FC layers for classification.

        - The GRU expects a tensor of the shape $(B, N, n_features_asf)$, but in the
        forward function of the model, we reshape the input tensor of shape (B, N, C=1, T, F)
        to a tensor of shape (B*N, C=1, T, F). We then need to reshape it back to (B, N, n_features_asf)
        before passing it to the GRU.
        - n_features_asf is the number of features at the output of the ResNet module, which is 2304.

        The input vector of the GRUResNetBasicHead is of shape (B*N, 1, 1, n_features_asf). We perform
        the following operations:

        1. Squeeze it: (B*N, 1, 1, n_features_asf) -> (B*N, n_features_asf)
        2. View it: (B*N, n_features_asf) -> (B, N, n_features_asf)
        3. Pass it through the GRU. Output of the GRU is (B, N, D * gru_hidden_size) = (B, N, 2 * 512)
        4. Reshape it back: (B, N, 1024) -> (B*N, 1024)
        5. Unsqueeze it to add the channel dimension: (B*N, 1024) -> (B*N, 1, 1, 1024)
        6. Project it back to CLIP embedding space: (B*N, 1, 1, 1024) -> (B*N, 1, 1, 512)
        7. Project it back to the number of features of the input:
           (B*N, 1, 1, 512) -> (B*N, 1, 1, n_features_asf)
        """
        # Reshape noun_embeddings to be (2 * num_gpu_layers, batch_size, embedding_size)
        B, N = initial_batch_shape
        F = x.shape[-1]
        D = 2 if self.gru.bidirectional else 1

        # (B*N, 1, 1, n_features_asf) -> (B*N, n_features_asf)
        # Squeeze all dimensions except the batch dimension (first)
        x = x.squeeze(1).squeeze(1)

        # (B*N, n_features_asf) -> (B, N, n_features_asf)
        x = x.view(B, N, F)

        # Pass the transformed batch through the GRU
        # (B, N, D * gru_hidden_size) = (B, N, 2 * 512)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        if not self.only_action_recognition:
            h_0 = (
                noun_embeddings.unsqueeze(0).repeat(D * self.gru_num_layers, 1, 1)
                if noun_embeddings is not None
                else None
            )

            x, _ = self.gru(x, hx=h_0)
        else:
            x, _ = self.gru(x)

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # (B, N, 1024) -> (B*N, 1024)
        x = x.reshape(B * N, D * self.gru.hidden_size)
        x = x.unsqueeze(1).unsqueeze(1)

        x = self.projection_to_dim_in(x)

        return x

    def project_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the GRU output to the state vector space.
        """
        x_min_1 = self.projection_min_1(x)  # (B, N, P)
        x_0 = self.projection_0(x)  # (B, N, P)
        x_1 = self.projection_1(x)  # (B, N, P)

        # Concatenate them into a single tensor of size (B, N, 3, P)
        x = torch.cat([x_min_1, x_0, x_1], dim=2)

        return x
