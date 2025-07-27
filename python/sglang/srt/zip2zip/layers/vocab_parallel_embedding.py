import logging
from typing import Optional, List

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from zip2zip.nn.encoders.base import BaseEncoder

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = logging.getLogger(__name__)


class Zip2ZipVocabParallelEmbedding(torch.nn.Module):
    def __init__(
        self,
        embed_tokens: VocabParallelEmbedding,
        input_encoder: BaseEncoder,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        assert embed_tokens.quant_config is None, "zip2zip doesn't support quantization"

        self.pad_token_id = pad_token_id
        self.input_encoder = input_encoder
        self.embed_tokens = embed_tokens

    def get_sharded_to_full_mapping(self) -> Optional[List[int]]:
        return self.embed_tokens.get_sharded_to_full_mapping()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        self.embed_tokens.weight_loader(param, loaded_weight)

    def forward(
        self, input_: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        ivs = self.input_encoder.compression_config.initial_vocab_size
        base_token_mask = input_ < ivs
        hyper_token_mask = ~base_token_mask
        base_input_ids = input_ * base_token_mask.long()
        hyper_input_ids = (input_ - ivs) * hyper_token_mask.long()

        indices_mask = forward_batch.updates_indices != -1

        forward_batch.hyper_embedding_weight[
            forward_batch.updates_indices * indices_mask
        ] = self.input_encoder(
            forward_batch.updates, self.embed_tokens.weight, self.pad_token_id
        ) * indices_mask.unsqueeze(-1)

        batch_offsets = torch.arange(
            input_.size(0), device=input_.device, dtype=torch.long
        ).unsqueeze(-1).expand_as(input_) * forward_batch.hyper_embedding_weight.size(1)

        hyper_input_ids += batch_offsets
        base_embedding = self.embed_tokens(
            base_input_ids, forward_batch
        ) * base_token_mask.unsqueeze(-1)
        hyper_embedding = F.embedding(
            hyper_input_ids,
            forward_batch.hyper_embedding_weight.view(
                -1, self.embed_tokens.embedding_dim
            ),
        ) * hyper_token_mask.unsqueeze(-1)

        return base_embedding + hyper_embedding

    def extra_repr(self) -> str:
        return self.embed_tokens.extra_repr()
