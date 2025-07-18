from typing import Tuple
import torch
import torch.nn as nn

from zip2zip import Zip2ZipConfig
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip.nn.encoders.config import EncoderConfigType
from zip2zip_compression import CompressionConfig, CodebookManager


class Zip2ZipManager:
    def __init__(
        self,
        zip2zip_config: Zip2ZipConfig[EncoderConfigType],
        compression_config: CompressionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.zip2zip_config = zip2zip_config
        self.compression_config = compression_config
        self.codebook_manager = CodebookManager(compression_config)

        self.input_encoder, self.output_encoder = self.get_encoders()

        self.updates = None
        self.indices = None
        self.hyper_embedding_weight = None
        self.hyper_linear_weight = None

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

    def reset(self) -> None:
        self.hyper_embedding_weight = None
        self.hyper_linear_weight = None
