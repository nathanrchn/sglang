import logging
from typing import Optional, List

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from zip2zip.nn.encoders.base import BaseEncoder

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.zip2zip.hyper_weight_pool import update_hyper_weights_pooled

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

        # Update hyper embedding weights using the pool system
        if (
            forward_batch.updates is not None
            and forward_batch.updates_indices is not None
            and forward_batch.hyper_weight_pool is not None
            and forward_batch.updates.numel() > 0
        ):
            # Reshape updates for encoder: from [total_updates, max_subtokens] to [batch_size, max_updates, max_subtokens]
            batch_size = forward_batch.batch_size
            total_updates = forward_batch.updates.shape[0]
            max_subtokens = forward_batch.updates.shape[1]
            
            if total_updates > 0:
                # Find max updates per batch for reshaping
                req_update_counts = []
                for i in range(batch_size):
                    start_idx = forward_batch.req_to_update_mapping[i, 0]
                    end_idx = forward_batch.req_to_update_mapping[i, 1]
                    req_update_counts.append(end_idx - start_idx)
                
                max_updates_per_batch = max(req_update_counts) if req_update_counts else 0
                
                if max_updates_per_batch > 0:
                    # Create reshaped tensor [batch_size, max_updates, max_subtokens]
                    reshaped_updates = torch.full(
                        (batch_size, max_updates_per_batch, max_subtokens),
                        fill_value=self.pad_token_id,
                        device=forward_batch.updates.device,
                        dtype=forward_batch.updates.dtype
                    )
                    
                    # Fill in the actual updates
                    for i in range(batch_size):
                        start_idx = forward_batch.req_to_update_mapping[i, 0]
                        end_idx = forward_batch.req_to_update_mapping[i, 1]
                        num_updates = end_idx - start_idx
                        if num_updates > 0:
                            reshaped_updates[i, :num_updates] = forward_batch.updates[start_idx:end_idx]
                    
                    # Compute embedding updates using the input encoder
                    embedding_updates = self.input_encoder(
                        reshaped_updates, self.embed_tokens.weight, self.pad_token_id
                    )
                    
                    # Flatten back for the kernel
                    flat_embedding_updates = []
                    for i in range(batch_size):
                        start_idx = forward_batch.req_to_update_mapping[i, 0]
                        end_idx = forward_batch.req_to_update_mapping[i, 1]
                        num_updates = end_idx - start_idx
                        if num_updates > 0:
                            flat_embedding_updates.append(embedding_updates[i, :num_updates])
                    
                    if flat_embedding_updates:
                        flat_embedding_updates = torch.cat(flat_embedding_updates, dim=0)
                        
                        # Create combined updates (embedding + zeros for linear part)
                        hidden_size = flat_embedding_updates.shape[-1]

                        combined_updates = torch.zeros(
                            (flat_embedding_updates.shape[0], 2 * hidden_size),
                            device=forward_batch.updates.device,
                            dtype=flat_embedding_updates.dtype,
                        )
                        combined_updates[:, :hidden_size] = flat_embedding_updates
                        # Linear part (second half) remains zeros for embedding layer

                        # Update the pool using the efficient Triton kernel
                        update_hyper_weights_pooled(
                            forward_batch.hyper_weight_pool.get_embedding_buffer(),
                            forward_batch.hyper_weight_pool.get_linear_buffer(),
                            combined_updates,
                            forward_batch.updates_indices,
                            forward_batch.hyper_weight_pool_indices,
                            forward_batch.req_to_update_mapping,
                        )

        # Get base embeddings
        base_embedding = self.embed_tokens(
            base_input_ids, forward_batch
        ) * base_token_mask.unsqueeze(-1)

        # Get hyper embeddings from the pool
        if forward_batch.hyper_weight_pool is not None:
            # Create hyper embedding tensor
            hyper_embedding = torch.zeros_like(base_embedding)
            
            if hyper_token_mask.any():
                # Use pre-computed batch indices (CUDA graph compatible)
                if forward_batch.token_to_batch_indices is not None:
                    batch_indices = forward_batch.token_to_batch_indices
                else:
                    # Fallback for compatibility (should not happen in normal execution)
                    num_tokens = input_.shape[0]
                    batch_indices = torch.zeros(num_tokens, device=input_.device, dtype=torch.long)

                # Get pool indices for each token
                pool_slots = forward_batch.hyper_weight_pool_indices[batch_indices]

                hyper_tokens = hyper_input_ids[hyper_token_mask]
                hyper_pool_slots = pool_slots[hyper_token_mask]

                # Lookup embeddings from the pool
                embedding_buffer = forward_batch.hyper_weight_pool.get_embedding_buffer()

                for i, (token_id, pool_slot) in enumerate(
                    zip(hyper_tokens, hyper_pool_slots)
                ):
                    if token_id < embedding_buffer.shape[1]:  # Check bounds
                        hyper_embedding[hyper_token_mask][i] = embedding_buffer[
                            pool_slot, token_id
                        ]
        else:
            hyper_embedding = torch.zeros_like(base_embedding)

        return base_embedding + hyper_embedding

    def extra_repr(self) -> str:
        return self.embed_tokens.extra_repr()
