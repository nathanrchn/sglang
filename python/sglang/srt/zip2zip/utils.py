from typing import Union

import torch
import triton
import triton.language as tl
from zip2zip_compression import LZWCompressor
from zip2zip import Zip2ZipTokenizer, Zip2ZipConfig
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def get_zip2zip_tokenizer(
    zip2zip_path: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Zip2ZipTokenizer:
    config = Zip2ZipConfig.from_pretrained(zip2zip_path)

    tokenizer = Zip2ZipTokenizer(config, tokenizer)
    del config

    return tokenizer


def get_lzw_compressor(
    zip2zip_path: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> LZWCompressor:
    config = Zip2ZipConfig.from_pretrained(zip2zip_path)
    compression_config = config.compression

    compressor = LZWCompressor(
        initial_vocab_size=compression_config.initial_vocab_size,
        max_codebook_size=compression_config.max_codebook_size,
        max_subtokens=compression_config.max_subtokens,
        pad_token_id=tokenizer.pad_token_id,
        disabled_ids=compression_config.disabled_ids,
    )
    del config

    return compressor


@triton.jit
def hyper_weight_update_kernel(
    hyper_weight_ptr,
    updates_ptr,
    updates_indices_ptr,
    hidden_dim,
    n_updates,
    n_updates_per_request,
    hyper_vocab_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # offsets for the updates dimension (from 0 to n_updates-1)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_updates

    # offsets for the hidden dimension
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < hidden_dim

    # --- Load source data from updates tensor ---
    source_offsets = offsets_m[:, None] * hidden_dim + offsets_n[None, :]
    mask_load = mask_m[:, None] & mask_n[None, :]
    
    values = tl.load(
        updates_ptr + source_offsets,
        mask=mask_load,
        other=0.0,
    )

    # --- Store to hyper_weight tensor (scatter) ---
    dest_indices = tl.load(updates_indices_ptr + offsets_m, mask=mask_m, other=0)
    
    # Decompose flattened update index `offsets_m` to get batch index.
    bs_idx = offsets_m // n_updates_per_request
    
    # Calculate destination offsets
    dest_offsets = (
        bs_idx[:, None] * hyper_vocab_size * hidden_dim
        + dest_indices[:, None] * hidden_dim
        + offsets_n[None, :]
    )

    tl.store(
        hyper_weight_ptr + dest_offsets,
        values,
        mask=mask_load,
    )


def hyper_weight_update(
    hyper_weight: torch.Tensor,
    updates: torch.Tensor,
    updates_indices: torch.Tensor,
):
    hyper_weight_size = hyper_weight.size()
    updates_size = updates.size()

    hyper_weight_contiguous = hyper_weight.contiguous()
    updates_contiguous = updates.contiguous()
    updates_indices_contiguous = updates_indices.contiguous()

    grid = lambda meta: (
        triton.cdiv(updates_size[0], meta['BLOCK_SIZE_M']),
        triton.cdiv(hyper_weight_size[1], meta['BLOCK_SIZE_N']),
    )

    return hyper_weight_update_kernel[grid](
        hyper_weight_contiguous,
        updates_contiguous,
        updates_indices_contiguous,
        hyper_weight_size[1],
        updates_size[0],
        updates_size[1],
        hyper_weight_size[0],
        BLOCK_SIZE_M=1024,
        BLOCK_SIZE_N=1024,
    )
