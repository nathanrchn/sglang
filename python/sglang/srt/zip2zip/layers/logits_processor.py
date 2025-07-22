import logging
from typing import Optional, Union

import torch
from zip2zip.nn.encoders.base import BaseEncoder

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    LogitsMetadata,
    ForwardBatch,
)

logger = logging.getLogger(__name__)


class Zip2ZipLogitsProcessor(torch.nn.Module):
    def __init__(self, logits_processor: LogitsProcessor, output_encoder: BaseEncoder) -> None:
        super().__init__()

        self.output_encoder = output_encoder
        self.logits_processor = logits_processor

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        aux_hidden_states: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        logger.info(f"Zip2ZipLogitsProcessor input_ids.shape: {input_ids.shape}")
        assert isinstance(
            logits_metadata, ForwardBatch
        ), "logits_metadata must be a ForwardBatch object"

        forward_batch = logits_metadata
        logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)

        assert (
            not logits_metadata.return_logprob
        ), "zip2zip implementation doesn't yet support return_logprob"

        output: LogitsProcessorOutput = self.logits_processor(
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states,
        )

        return output
