from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from zip2zip.config import Zip2ZipConfig
from zip2zip.nn.encoders.base import BaseEncoder

from sglang.srt.layers.logits_processor import fused_softcap
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.utils import dump_to_file, use_intel_amx_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    LogitsMetadata,
    ForwardBatch,
)
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    dp_gather_replicate,
    dp_scatter,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.zip2zip.hyper_weight_pool import update_hyper_weights_pooled

logger = logging.getLogger(__name__)


class Zip2ZipLogitsProcessor(torch.nn.Module):
    def __init__(
        self,
        config,
        zip2zip_config: Zip2ZipConfig,
        output_encoder: BaseEncoder,
        pad_token_id: int,
        skip_all_gather: bool = False,
        logit_scale: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id
        self.zip2zip_config = zip2zip_config
        self.output_encoder = output_encoder
        self.logit_scale = logit_scale
        self.use_attn_tp_group = global_server_args_dict["enable_dp_lm_head"]
        if self.use_attn_tp_group:
            self.attn_tp_size = get_attention_tp_size()
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and self.attn_tp_size > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = False
        else:
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and get_tensor_model_parallel_world_size() > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = (
                self.do_tensor_parallel_all_gather and get_attention_dp_size() != 1
            )
        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None

        self.debug_tensor_dump_output_folder = global_server_args_dict.get(
            "debug_tensor_dump_output_folder", None
        )

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        aux_hidden_states: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        assert isinstance(
            logits_metadata, ForwardBatch
        ), "zip2zip requires logits_metadata to be a ForwardBatch object"

        forward_batch = logits_metadata
        logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)

        # Get the last hidden states and last logits for the next token prediction
        if (
            logits_metadata.forward_mode.is_decode_or_idle()
            or logits_metadata.forward_mode.is_target_verify()
        ):
            pruned_states = hidden_states
            if aux_hidden_states is not None:
                aux_pruned_states = [hidden for hidden in aux_hidden_states]
            sample_indices = None
            input_logprob_indices = None
        elif (
            logits_metadata.forward_mode.is_extend()
            and not logits_metadata.extend_return_logprob
        ):
            # Prefill without input logprobs.
            if logits_metadata.padded_static_len < 0:
                last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            else:
                # If padding_static length is 5 and extended_seq_lens is [2, 3],
                # then our batch looks like [t00, t01, p, p, p, t10, t11, t12, p, p]
                # and this retrieves t01 and t12, which are the valid last tokens
                idx = torch.arange(
                    len(logits_metadata.extend_seq_lens),
                    device=logits_metadata.extend_seq_lens.device,
                )
                last_index = (
                    idx * logits_metadata.padded_static_len
                    + logits_metadata.extend_seq_lens
                    - 1
                )
            pruned_states = hidden_states[last_index]
            if aux_hidden_states is not None:
                aux_pruned_states = [hidden[last_index] for hidden in aux_hidden_states]
            sample_indices = None
            input_logprob_indices = None
        else:
            # Input logprobs are required.
            # Find 3 different indices.
            # 1. pruned_states: hidden states that we want logprobs from.
            # 2. sample_indices: Indices that have sampled tokens.
            # 3. input_logprob_indices: Indices that have input logprob tokens.
            sample_index_pt = -1
            sample_indices = []
            input_logprob_indices_pt = 0
            input_logprob_indices = []
            pt, pruned_states = 0, []
            for extend_logprob_start_len, extend_len in zip(
                logits_metadata.extend_logprob_start_lens_cpu,
                logits_metadata.extend_seq_lens_cpu,
            ):
                # It can happen in chunked prefill. We still need to sample 1 token,
                # But we don't want to include it in input logprob.
                if extend_len == extend_logprob_start_len:
                    start_len = extend_logprob_start_len - 1
                else:
                    start_len = extend_logprob_start_len

                # We always need at least 1 token to sample because that's required
                # by a caller.
                assert extend_len > start_len
                pruned_states.append(hidden_states[pt + start_len : pt + extend_len])
                pt += extend_len
                sample_index_pt += extend_len - start_len
                sample_indices.append(sample_index_pt)
                input_logprob_indices.extend(
                    [
                        input_logprob_indices_pt + i
                        for i in range(extend_len - extend_logprob_start_len)
                    ]
                )
                input_logprob_indices_pt += extend_len - start_len

            pruned_states = torch.cat(pruned_states)
            sample_indices = torch.tensor(
                sample_indices, device=pruned_states.device, dtype=torch.int64
            )
            input_logprob_indices = torch.tensor(
                input_logprob_indices, device=pruned_states.device, dtype=torch.int64
            )

        # Compute logits for both input and sampled tokens.
        logits = self._get_logits(
            pruned_states, lm_head, forward_batch, logits_metadata
        )
        sampled_logits = (
            logits[sample_indices] if sample_indices is not None else logits
        )

        if self.debug_tensor_dump_output_folder:
            assert (
                not self.do_tensor_parallel_all_gather
                or get_local_attention_dp_size() == 1
            ), "dp attention + sharded lm_head doesn't support full logits"
            full_logits = self._get_logits(
                hidden_states, lm_head, forward_batch, logits_metadata
            )
            dump_to_file(self.debug_tensor_dump_output_folder, "logits", full_logits)

        hidden_states_to_store: Optional[torch.Tensor] = None
        if logits_metadata.capture_hidden_mode.need_capture():
            if logits_metadata.capture_hidden_mode.is_full():
                if aux_hidden_states is not None:
                    aux_hidden_states = torch.cat(aux_hidden_states, dim=-1)
                    hidden_states_to_store = aux_hidden_states
                else:
                    hidden_states_to_store = hidden_states
            elif logits_metadata.capture_hidden_mode.is_last():
                # Get the last token hidden states. If sample_indices is None,
                # pruned states only contain the last tokens already.
                if aux_hidden_states is not None:
                    aux_pruned_states = torch.cat(aux_pruned_states, dim=-1)
                    hidden_states_to_store = (
                        aux_pruned_states[sample_indices]
                        if sample_indices is not None
                        else aux_pruned_states
                    )
                else:
                    hidden_states_to_store = (
                        pruned_states[sample_indices]
                        if sample_indices is not None
                        else pruned_states
                    )
            else:
                assert False, "Should never reach"

        if not logits_metadata.extend_return_logprob:
            # Decode mode or extend mode without return_logprob.
            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                hidden_states=hidden_states_to_store,
            )
        else:
            input_logprobs = logits[input_logprob_indices]
            del hidden_states, logits

            # Normalize the logprob w/o temperature, top-p
            pruned_lens = torch.tensor(
                logits_metadata.extend_logprob_pruned_lens_cpu,
                device=input_logprobs.device,
            )
            if logits_metadata.temp_scaled_logprobs:
                logits_metadata.temperature = torch.repeat_interleave(
                    logits_metadata.temperature.view(-1),
                    pruned_lens,
                ).view(-1, 1)
            if logits_metadata.top_p_normalized_logprobs:
                logits_metadata.top_p = torch.repeat_interleave(
                    logits_metadata.top_p,
                    pruned_lens,
                )
            input_logprobs = self.compute_temp_top_p_normalized_logprobs(
                input_logprobs, logits_metadata
            )

            # Get the logprob of top-k tokens
            if logits_metadata.extend_return_top_logprob:
                (
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                ) = self.get_top_logprobs(input_logprobs, logits_metadata)
            else:
                input_top_logprobs_val = input_top_logprobs_idx = None

            # Get the logprob of given token id
            if logits_metadata.extend_token_ids_logprob:
                (
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                ) = self.get_token_ids_logprobs(input_logprobs, logits_metadata)
            else:
                input_token_ids_logprobs_val = input_token_ids_logprobs_idx = None

            input_token_logprobs = input_logprobs[
                torch.arange(input_logprobs.shape[0], device=input_logprobs.device),
                logits_metadata.extend_input_logprob_token_ids_gpu,
            ]

            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                input_token_logprobs=input_token_logprobs,
                input_top_logprobs_val=input_top_logprobs_val,
                input_top_logprobs_idx=input_top_logprobs_idx,
                hidden_states=hidden_states_to_store,
                input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get logits from hidden_states.

        If sampled_logits_only is True, it means hidden_states only contain the
        last position (e.g., extend without input logprobs). The caller should
        guarantee the given hidden_states follow this constraint.
        """
        if self.do_tensor_parallel_all_gather_dp_attn:
            logits_metadata.compute_dp_attention_metadata(hidden_states)
            hidden_states, local_hidden_states = (
                torch.empty_like(logits_metadata.gathered_buffer),
                hidden_states,
            )
            dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)

        ivs = self.zip2zip_config.compression.initial_vocab_size
        mcs = self.zip2zip_config.compression.max_codebook_size
        num_tokens = hidden_states.shape[0]

        # Create logits tensor with correct shape: (num_tokens, vocab_size + mcs)
        logits = torch.empty(
            num_tokens,
            self.config.vocab_size + mcs,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Update hyper linear weights using the pool system
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
                    
                    # Compute linear updates using the output encoder
                    linear_updates = self.output_encoder(
                        reshaped_updates, lm_head.weight, self.pad_token_id
                    )
                    
                    # Flatten back for the kernel
                    flat_linear_updates = []
                    for i in range(batch_size):
                        start_idx = forward_batch.req_to_update_mapping[i, 0]
                        end_idx = forward_batch.req_to_update_mapping[i, 1]
                        num_updates = end_idx - start_idx
                        if num_updates > 0:
                            flat_linear_updates.append(linear_updates[i, :num_updates])
                    
                    if flat_linear_updates:
                        flat_linear_updates = torch.cat(flat_linear_updates, dim=0)
                        
                        # Create combined updates (zeros for embedding + linear part)
                        hidden_size = forward_batch.hyper_weight_pool.hidden_size

                        combined_updates = torch.zeros(
                            (flat_linear_updates.shape[0], 2 * hidden_size),
                            device=forward_batch.updates.device,
                            dtype=flat_linear_updates.dtype,
                        )
                        # Embedding part (first half) remains zeros for logits processor
                        combined_updates[:, hidden_size:] = flat_linear_updates

                        # Update the pool using the efficient Triton kernel
                        update_hyper_weights_pooled(
                            forward_batch.hyper_weight_pool.get_embedding_buffer(),
                            forward_batch.hyper_weight_pool.get_linear_buffer(),
                            combined_updates,
                            forward_batch.updates_indices,
                            forward_batch.hyper_weight_pool_indices,
                            forward_batch.req_to_update_mapping,
                        )

        # Compute base logits
        if hasattr(lm_head, "weight"):
            if use_intel_amx_backend(lm_head):
                output = torch.ops.sgl_kernel.weight_packed_linear(
                    hidden_states.to(lm_head.weight.dtype),
                    lm_head.weight,
                    None,  # bias
                    True,  # is_vnni
                )
            else:
                output = torch.matmul(
                    hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                )
        else:
            # GGUF models
            # TODO: use weight_packed_linear for GGUF models
            output = lm_head.quant_method.apply(lm_head, hidden_states, embedding_bias)

        # Fill in the logits tensor
        logits[:, :ivs] = output[:, :ivs]

        # Compute hyper logits using the pool system
        if forward_batch.hyper_weight_pool is not None:
            # Use pre-computed batch indices (CUDA graph compatible)
            if forward_batch.token_to_batch_indices is not None:
                batch_indices = forward_batch.token_to_batch_indices
            else:
                # Fallback for compatibility (should not happen in normal execution)
                batch_indices = torch.zeros(num_tokens, device=hidden_states.device, dtype=torch.long)

            # Get pool indices for each token
            pool_slots = forward_batch.hyper_weight_pool_indices[batch_indices]

            # Get linear buffer from pool
            linear_buffer = forward_batch.hyper_weight_pool.get_linear_buffer()

            # Compute hyper logits efficiently
            hyper_logits = torch.zeros(
                (num_tokens, mcs),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            for i in range(num_tokens):
                pool_slot = pool_slots[i]
                # Matrix multiply: hidden_states[i] @ linear_buffer[pool_slot].T
                hyper_logits[i] = torch.matmul(
                    hidden_states[i : i + 1], linear_buffer[pool_slot].T
                ).squeeze(0)

            logits[:, ivs : ivs + mcs] = hyper_logits
        else:
            # No hyper weights, fill with zeros
            logits[:, ivs : ivs + mcs] = 0

        logits[:, ivs + mcs :] = output[:, ivs:]

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        if self.do_tensor_parallel_all_gather:
            if self.use_attn_tp_group:
                global_logits = torch.empty(
                    (self.config.vocab_size + mcs, logits.shape[0]),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                global_logits = global_logits.T
                attn_tp_all_gather(
                    list(global_logits.tensor_split(self.attn_tp_size, dim=-1)), logits
                )
                logits = global_logits
            else:
                logits = tensor_model_parallel_all_gather(logits)

        if self.do_tensor_parallel_all_gather_dp_attn:
            logits, global_logits = (
                torch.empty(
                    (local_hidden_states.shape[0], logits.shape[1]),
                    device=logits.device,
                    dtype=logits.dtype,
                ),
                logits,
            )
            dp_scatter(logits, global_logits, logits_metadata)

        logits = logits[:, : self.config.vocab_size + mcs].float()

        if self.final_logit_softcapping:
            fused_softcap(logits, self.final_logit_softcapping)

        return logits

    @staticmethod
    def get_top_logprobs(all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata):
        max_k = max(logits_metadata.top_logprobs_nums)
        ret = all_logprobs.topk(max_k, dim=1)
        values = ret.values.tolist()
        indices = ret.indices.tolist()

        input_top_logprobs_val, input_top_logprobs_idx = [], []

        pt = 0
        for k, pruned_len in zip(
            logits_metadata.top_logprobs_nums,
            logits_metadata.extend_logprob_pruned_lens_cpu,
        ):
            if pruned_len <= 0:
                input_top_logprobs_val.append([])
                input_top_logprobs_idx.append([])
                continue

            input_top_logprobs_val.append(
                [values[pt + j][:k] for j in range(pruned_len)]
            )
            input_top_logprobs_idx.append(
                [indices[pt + j][:k] for j in range(pruned_len)]
            )
            pt += pruned_len

        return input_top_logprobs_val, input_top_logprobs_idx

    @staticmethod
    def get_token_ids_logprobs(
        all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata
    ):
        input_token_ids_logprobs_val, input_token_ids_logprobs_idx = [], []
        pt = 0
        for token_ids, pruned_len in zip(
            logits_metadata.token_ids_logprobs,
            logits_metadata.extend_logprob_pruned_lens_cpu,
        ):
            if pruned_len <= 0:
                input_token_ids_logprobs_val.append([])
                input_token_ids_logprobs_idx.append([])
                continue

            input_token_ids_logprobs_val.append(
                [all_logprobs[pt + j, token_ids].tolist() for j in range(pruned_len)]
            )
            input_token_ids_logprobs_idx.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len

        return input_token_ids_logprobs_val, input_token_ids_logprobs_idx

    @staticmethod
    def compute_temp_top_p_normalized_logprobs(
        last_logits: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> torch.Tensor:
        """
        compute logprobs for the output token from the given logits.

        Returns:
            torch.Tensor: logprobs from logits
        """
        # Scale logits if temperature scaling is enabled
        if logits_metadata.temp_scaled_logprobs:
            last_logits = last_logits / logits_metadata.temperature

        # Normalize logprobs if top_p normalization is enabled
        # NOTE: only normalize logprobs when top_p is set and not equal to 1.0
        if (
            logits_metadata.top_p_normalized_logprobs
            and (logits_metadata.top_p != 1.0).any()
        ):
            from sglang.srt.layers.sampler import top_p_normalize_probs_torch

            probs = torch.softmax(last_logits, dim=-1)
            del last_logits
            probs = top_p_normalize_probs_torch(probs, logits_metadata.top_p)
            return torch.log(probs)
        else:
            return torch.nn.functional.log_softmax(last_logits, dim=-1)

    @staticmethod
    def from_logits_processor(
        logits_processor: LogitsProcessor,
    ) -> Zip2ZipLogitsProcessor:
        return Zip2ZipLogitsProcessor(
            config=logits_processor.config,
            logit_scale=logits_processor.logit_scale,
        )
