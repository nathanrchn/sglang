import os
from typing import Tuple, Optional

import torch
from zip2zip import Zip2ZipConfig
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip.constants import SAFETENSORS_ENCODERS_NAME
from zip2zip.nn.encoders.config import EncoderConfigType
from zip2zip_compression import CompressionConfig, CodebookManager

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Zip2ZipManager:
    def __init__(
        self,
        zip2zip_config: Zip2ZipConfig[EncoderConfigType],
        compression_config: CompressionConfig,
        dtype: torch.dtype,
        device: torch.device,
        zip2zip_path: str,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.zip2zip_config = zip2zip_config
        self.compression_config = compression_config
        self.codebook_manager = CodebookManager(compression_config)

        self.input_encoder, self.output_encoder = self.get_encoders()
        self.load_pretrained_encoders(zip2zip_path)

        self.forward_batch: Optional[ForwardBatch] = None

    def get_encoders(self) -> Tuple[BaseEncoder, BaseEncoder]:
        input_encoder = BaseEncoder.from_config(
            self.zip2zip_config.encoder, self.zip2zip_config.compression
        ).to(self.device, self.dtype)

        if self.zip2zip_config.encoder.tie_encoders:
            return input_encoder, None
        else:
            output_encoder = BaseEncoder.from_config(
                self.zip2zip_config.encoder, self.zip2zip_config.compression
            ).to(self.device, self.dtype)
            return input_encoder, output_encoder

    def get_hyper_embedding_weight(
        self, ids: torch.Tensor, base_weight: torch.Tensor
    ) -> torch.Tensor:
        return self.hyper_embedding_weight

    def get_hyper_linear_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        return self.hyper_linear_weight

    def load_pretrained_encoders(
        self,
        zip2zip_path: str,
    ) -> None:
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

        self.input_encoder.load_state_dict(input_encoder_state_dict)
        if self.output_encoder is not None:
            self.output_encoder.load_state_dict(output_encoder_state_dict)
