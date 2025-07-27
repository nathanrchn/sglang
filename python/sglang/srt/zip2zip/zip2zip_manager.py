import os
from typing import List, Tuple, Optional

import torch
from zip2zip import Zip2ZipConfig
from transformers import AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip.constants import SAFETENSORS_ENCODERS_NAME
from zip2zip.nn.encoders.config import EncoderConfigType
from zip2zip_compression import CompressionConfig, CodebookManager

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.zip2zip.layers.logits_processor import Zip2ZipLogitsProcessor
from sglang.srt.zip2zip.layers.vocab_parallel_embedding import (
    Zip2ZipVocabParallelEmbedding,
)


class Zip2ZipManager:
    def __init__(
        self,
        base_model: torch.nn.Module,
        model_config: ModelConfig,
        zip2zip_path: str,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.base_model = base_model
        self.model_config = model_config
        self.config: Zip2ZipConfig[EncoderConfigType] = model_config.zip2zip_config

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)
        self.codebook_manager = CodebookManager(
            config=CompressionConfig(
                initial_vocab_size=self.config.compression.initial_vocab_size,
                max_codebook_size=self.config.compression.max_codebook_size,
                max_subtokens=self.config.compression.max_subtokens,
                pad_token_id=tokenizer.pad_token_id,
                disabled_ids=self.config.compression.disabled_ids,
            )
        )

        input_encoder, output_encoder = self.get_pretrained_encoders(zip2zip_path)

        assert hasattr(
            self.base_model.model, "embed_tokens"
        ), "the model needs to have an model.embed_tokens attribute to use zip2zip"
        assert hasattr(
            self.base_model, "logits_processor"
        ), "the model needs to have a logits_processor attribute to use zip2zip"

        self.base_model.model.embed_tokens = Zip2ZipVocabParallelEmbedding(
            self.base_model.model.embed_tokens, input_encoder
        )

        lp = self.base_model.logits_processor
        self.base_model.logits_processor = Zip2ZipLogitsProcessor(
            config=lp.config,
            zip2zip_config=self.config,
            output_encoder=output_encoder,
            pad_token_id=tokenizer.pad_token_id,
            skip_all_gather=False,
            logit_scale=lp.logit_scale,
        )

    def update_compression_states(
        self, batch: ModelWorkerBatch
    ) -> Tuple[List[List[int]], List[List[int]]]:
        return self.codebook_manager.update_codebooks(
            batch.input_ids.tolist(), batch.compression_states, True
        )

    def get_pretrained_encoders(
        self, zip2zip_path: str
    ) -> Tuple[BaseEncoder, Optional[BaseEncoder]]:
        input_encoder = BaseEncoder.from_config(
            self.config.encoder, self.config.compression
        ).to(self.device, self.dtype)

        output_encoder = None
        if not self.config.encoder.tie_encoders:
            output_encoder = BaseEncoder.from_config(
                self.config.encoder, self.config.compression
            ).to(self.device, self.dtype)

        if os.path.isfile(os.path.join(zip2zip_path, SAFETENSORS_ENCODERS_NAME)):
            encoder_file = os.path.join(zip2zip_path, SAFETENSORS_ENCODERS_NAME)
        else:
            try:
                encoder_file = hf_hub_download(
                    zip2zip_path,
                    SAFETENSORS_ENCODERS_NAME,
                )
            except Exception as exc:
                raise ValueError(
                    f"Can't find '{SAFETENSORS_ENCODERS_NAME}' at '{zip2zip_path}'"
                ) from exc

        encoders_state_dict = load_file(encoder_file, device=self.device)

        input_encoder_state_dict = {}
        output_encoder_state_dict = {}

        for k, v in encoders_state_dict.items():
            if k.startswith("input_encoder."):
                input_encoder_state_dict[k.removeprefix("input_encoder.")] = v
            elif k.startswith("output_encoder."):
                output_encoder_state_dict[k.removeprefix("output_encoder.")] = v

        input_encoder.load_state_dict(input_encoder_state_dict)
        if output_encoder is not None:
            output_encoder.load_state_dict(output_encoder_state_dict)

        return input_encoder, output_encoder
