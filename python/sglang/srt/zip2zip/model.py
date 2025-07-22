import logging
from typing import Optional, List, Iterable, Tuple

import torch
from torch import nn
from zip2zip import Zip2ZipConfig

from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.zip2zip.manager import Zip2ZipManager
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

BaseModelConfig = None

logger = logging.getLogger(__name__)


class Zip2ZipModel(nn.Module):
    def __init__(
        self,
        base_model: LlamaForCausalLM,
        zip2zip_path: str,
    ) -> None:
        super().__init__()
        self.base_model = base_model

        self.dtype = self.base_model.dtype
        self.device = self.base_model.device
        self.config = Zip2ZipConfig.from_pretrained(zip2zip_path)

        self.manager = Zip2ZipManager(
            self.config, self.dtype, self.device, zip2zip_path
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        return self.model.forward(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            get_embedding,
            pp_proxy_tensors,
        )

    @property
    def start_layer(self):
        return self.base_model.start_layer

    @property
    def end_layer(self):
        return self.base_model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.base_model.get_input_embeddings()

    def get_hidden_dim(self, module_name):
        return self.base_model.get_hidden_dim(module_name)

    def get_module_name(self, name):
        return self.base_model.get_module_name(name)

    def get_module_name_from_weight_name(self, name):
        return self.base_model.get_module_name_from_weight_name(name)

    def get_num_params(self):
        return self.base_model.get_num_params()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return self.base_model.load_weights(weights)

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100, tp_size: int = 1
    ) -> Optional[torch.Tensor]:
        return self.base_model.get_weights_by_name(name, truncate_size, tp_size)

    def get_embed_and_head(self):
        return self.base_model.get_embed_and_head()

    def set_embed_and_head(self, embed, head):
        return self.base_model.set_embed_and_head(embed, head)

    def get_embed(self):
        return self.base_model.get_embed()

    def set_embed(self, embed):
        return self.base_model.set_embed(embed)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        return self.base_model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        return self.base_model.set_eagle3_layers_to_capture(layer_ids)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        return self.base_model.set_eagle3_layers_to_capture(layer_ids)
