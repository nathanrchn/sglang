from typing import Tuple
import torch
import torch.nn as nn

from zip2zip import Zip2ZipConfig
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip_compression import CompressionConfig, CodebookManager

class Zip2ZipManager:
    def __init__(self, zip2zip_config: Zip2ZipConfig, compression_config: CompressionConfig, dtype: torch.dtype, device: torch.device) -> None:
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

    def get_hyper_embedding_weight(self, ids: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        updates, indices = self.codebook_manager.update_codebooks(ids.tolist())

        self.updates = torch.tensor(
            updates,
            device=self.device,
            dtype=torch.long,
        ).view(ids.shape[0], -1, self.zip2zip_config.compression.max_subtokens)

        self.indices = indices
        if any(len(ui) > 0 for ui in self.updates_indices):
            new_weights = self.input_encoder(self.updates, base_weight, self.pad_token_id)

            for i, ui in enumerate(self.updates_indices):
                self.hyper_embedding_weight[i, ui] = new_weights[i, : len(ui)]

        return self.hyper_embedding_weight
    
    def get_hyper_linear_weight(self):
        return self.hyper_linear_weight

    def reset(self):
        self.hyper_embedding_weight = None
        self.hyper_linear_weight = None
