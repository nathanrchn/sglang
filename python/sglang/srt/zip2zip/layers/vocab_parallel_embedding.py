from typing import Optional, List

import torch
from torch.nn import Parameter
from zip2zip.nn.encoders.base import BaseEncoder

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding


class Zip2ZipVocabParallelEmbedding(torch.nn.Module):
    def __init__(
        self, embed_tokens: VocabParallelEmbedding, encoder: BaseEncoder
    ) -> None:
        super().__init__()
        assert embed_tokens.quant_config is None, "zip2zip doesn't support quantization"

        self.encoder = encoder
        self.embed_tokens = embed_tokens

    def get_sharded_to_full_mapping(self) -> Optional[List[int]]:
        return self.embed_tokens.get_sharded_to_full_mapping()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        self.embed_tokens.weight_loader(param, loaded_weight)

    def forward(
        self, input_: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        return self.embed_tokens(input_)

    def extra_repr(self) -> str:
        return self.embed_tokens.extra_repr()
