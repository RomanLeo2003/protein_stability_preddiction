from torchvision.ops import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer

model_name = "facebook/esm2_t33_650M_UR50D"


class MLPHead(MLP):
    def __init__(
        self,
        in_channels,
        dim_hidden,
        num_layers=3,
        norm_layer=None,
        dropout=0.0,
    ):
        hidden_channels = [dim_hidden] * (num_layers - 1) + [1]
        super(MLPHead, self).__init__(
            in_channels,
            hidden_channels,
            inplace=False,
            norm_layer=norm_layer,
            dropout=dropout,
        )


class LightAttention(nn.Module):
    def __init__(
        self,
        embeddings_dim=1024,
        output_dim=1280,
        dropout=0.25,
        kernel_size=9,
        conv_dropout: float = 0.25,
    ):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(
            x
        )  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(
            o * self.softmax(attention), dim=-1
        )  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class FFN(nn.Module):
    def __init__(
        self,
        embeddings_dim: int = 2560,
        output_dim: int = 1,
        hidden_dim: int = 512,
        n_hidden_layers: int = 3,
        dropout: float = 0.25,
    ):
        """
        Simple Feed forward model with default parameters like the network tha is ued in the SeqVec paper.
        Args:
            embeddings_dim: dimension of the input
            hidden_dim: dimension of the hidden layers
            output_dim: output dimension (number of classes that should be classified)
            n_hidden_layers: number of hidden layers (0 by default)
            dropout: dropout ratio of every layer
        """
        super(FFN, self).__init__()

        self.n_hidden_layers = n_hidden_layers
        self.input = nn.Sequential(
            nn.Linear(embeddings_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.hidden = nn.ModuleList()
        for _ in range(self.n_hidden_layers):
            self.hidden.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                )
            )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.input(x)
        for hidden_layer in self.hidden:
            o = hidden_layer(o)
        return self.output(o)


class ABYSSALModel(nn.Module):
    def __init__(self, esm_model_name, embed_dim, output_dim=1):
        super(ABYSSALModel, self).__init__()
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        self.esm_model.eval()
        self.light_attention = LightAttention(embeddings_dim=embed_dim)
        self.fc_block = FFN(embeddings_dim=embed_dim * 2, output_dim=output_dim)

    def forward(self, wt_tokens, mt_tokens, mask):

        wt_embedding = self.esm_model(**wt_tokens)["last_hidden_state"].transpose(1, 2)
        mt_embedding = self.esm_model(**mt_tokens)["last_hidden_state"].transpose(1, 2)

        attn_output1 = self.light_attention(wt_embedding, mask)
        attn_output2 = self.light_attention(mt_embedding, mask)

        concatenated = torch.cat((attn_output1, attn_output2), dim=-1)

        output = self.fc_block(concatenated)
        return output


# seq1 = tokenizer(
#     "ORIGINAL_SEQUENCE", return_tensors="pt", padding=True, truncation=True
# )
# seq2 = tokenizer("MUTATED_SEQUENCE", return_tensors="pt", padding=True, truncation=True)

# model = ABYSSALModel(esm_model, embed_dim=esm_model.config.hidden_size)
# output = model(seq1, seq2)
# print("Predicted ΔΔG:", output.item())
