"""
Memory pool for zip2zip hyper weights to support dynamic batching.

This module implements a memory pool system similar to KVCache but for
hyper embedding and linear weights used in zip2zip compression.
"""

from __future__ import annotations

import abc
import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)


GB = 1024 * 1024 * 1024


class BaseHyperWeightAllocator(abc.ABC):
    """Base allocator for hyper weight pool indices."""

    def __init__(
        self,
        size: int,
        max_codebook_size: int,
        dtype: torch.dtype,
        device: str,
        hyper_weight_pool: HyperWeightPool,
    ):
        self.size = size
        self.max_codebook_size = max_codebook_size
        self.dtype = dtype
        self.device = device
        self.hyper_weight_pool = hyper_weight_pool

    @abc.abstractmethod
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate hyper weight slots."""
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_indices: torch.Tensor) -> None:
        """Free hyper weight slots."""
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all allocations."""
        raise NotImplementedError()


class HyperWeightAllocator(BaseHyperWeightAllocator):
    """Simple allocator for hyper weight pool."""

    def __init__(
        self,
        size: int,
        max_codebook_size: int,
        dtype: torch.dtype,
        device: str,
        hyper_weight_pool: HyperWeightPool,
    ):
        super().__init__(size, max_codebook_size, dtype, device, hyper_weight_pool)
        self.free_slots = list(range(size))

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate consecutive hyper weight slots."""
        if need_size > len(self.free_slots):
            return None

        allocated_slots = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return torch.tensor(allocated_slots, dtype=torch.int64, device=self.device)

    def free(self, free_indices: torch.Tensor) -> None:
        """Free hyper weight slots."""
        if free_indices.numel() > 0:
            self.free_slots.extend(free_indices.cpu().tolist())

    def clear(self) -> None:
        """Clear all allocations."""
        self.free_slots = list(range(self.size))

    def available_size(self) -> int:
        """Get number of available slots."""
        return len(self.free_slots)


class HyperWeightPool:
    """Memory pool for hyper embedding and linear weights."""

    def __init__(
        self,
        size: int,
        max_codebook_size: int,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool = False,
    ):
        self.size = size
        self.max_codebook_size = max_codebook_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.device = device

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self._create_buffers()

        # Calculate memory usage
        embedding_size = self.get_embedding_size_bytes()
        linear_size = self.get_linear_size_bytes()
        total_size = embedding_size + linear_size

        logger.info(
            f"HyperWeight Pool allocated. #slots: {size}, "
            f"Embedding: {embedding_size / GB:.2f} GB, "
            f"Linear: {linear_size / GB:.2f} GB, "
            f"Total: {total_size / GB:.2f} GB"
        )
        self.mem_usage = total_size / GB

    def _create_buffers(self):
        """Create the hyper weight buffers."""
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # Hyper embedding weights: [size, max_codebook_size, hidden_size]
            self.embedding_buffer = torch.zeros(
                self.size,
                self.max_codebook_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

            # Hyper linear weights: [size, max_codebook_size, vocab_size]
            self.linear_buffer = torch.zeros(
                self.size,
                self.max_codebook_size,
                self.vocab_size,
                dtype=self.dtype,
                device=self.device,
            )

    def get_embedding_buffer(self) -> torch.Tensor:
        """Get the embedding buffer."""
        return self.embedding_buffer

    def get_linear_buffer(self) -> torch.Tensor:
        """Get the linear buffer."""
        return self.linear_buffer

    def get_embedding_size_bytes(self) -> int:
        """Get embedding buffer size in bytes."""
        return (
            self.size * self.max_codebook_size * self.hidden_size * self.dtype.itemsize
        )

    def get_linear_size_bytes(self) -> int:
        """Get linear buffer size in bytes."""
        return (
            self.size * self.max_codebook_size * self.vocab_size * self.dtype.itemsize
        )

    def clear_slot(self, slot_index: int):
        """Clear a specific slot in the pool."""
        self.embedding_buffer[slot_index].zero_()
        self.linear_buffer[slot_index].zero_()


@triton.jit
def hyper_weight_update_pooled_kernel(
    # Pool buffers
    embedding_pool_ptr,
    linear_pool_ptr,
    # Update data
    updates_ptr,
    updates_indices_ptr,
    # Pool indices for each request
    pool_indices_ptr,
    # Request mapping
    req_to_update_start_ptr,
    req_to_update_end_ptr,
    # Dimensions
    max_codebook_size,
    hidden_size,
    vocab_size,
    num_requests,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    """
    Triton kernel for updating hyper weights in the memory pool.

    This kernel processes updates for multiple requests and updates their
    corresponding slots in the hyper weight pool.
    """
    # Get request ID
    req_id = tl.program_id(0)
    if req_id >= num_requests:
        return

    # Get update range for this request
    update_start = tl.load(req_to_update_start_ptr + req_id)
    update_end = tl.load(req_to_update_end_ptr + req_id)
    num_updates = update_end - update_start

    if num_updates <= 0:
        return

    # Get pool slot for this request
    pool_slot = tl.load(pool_indices_ptr + req_id)

    # Process updates in blocks
    for update_offset in range(0, num_updates, 1):
        update_idx = update_start + update_offset
        if update_idx >= update_end:
            break

        # Get the codebook index for this update
        codebook_idx = tl.load(updates_indices_ptr + update_idx)
        if codebook_idx < 0 or codebook_idx >= max_codebook_size:
            continue

        # Update embedding weights
        for h_block in range(0, hidden_size, BLOCK_SIZE_H):
            h_start = h_block
            h_end = min(h_start + BLOCK_SIZE_H, hidden_size)
            h_mask = tl.arange(0, BLOCK_SIZE_H) < (h_end - h_start)

            # Load update values
            update_offset_emb = (
                update_idx * hidden_size + h_start + tl.arange(0, BLOCK_SIZE_H)
            )
            update_vals = tl.load(
                updates_ptr + update_offset_emb, mask=h_mask, other=0.0
            )

            # Store to embedding pool
            pool_offset_emb = (
                pool_slot * max_codebook_size * hidden_size
                + codebook_idx * hidden_size
                + h_start
                + tl.arange(0, BLOCK_SIZE_H)
            )
            tl.atomic_add(
                embedding_pool_ptr + pool_offset_emb, update_vals, mask=h_mask
            )

        # Update linear weights
        for v_block in range(0, vocab_size, BLOCK_SIZE_V):
            v_start = v_block
            v_end = min(v_start + BLOCK_SIZE_V, vocab_size)
            v_mask = tl.arange(0, BLOCK_SIZE_V) < (v_end - v_start)

            # Load update values (assuming updates contain both embedding and linear)
            # Offset by hidden_size to get linear part
            update_offset_lin = (
                update_idx * (hidden_size + vocab_size)
                + hidden_size
                + v_start
                + tl.arange(0, BLOCK_SIZE_V)
            )
            update_vals = tl.load(
                updates_ptr + update_offset_lin, mask=v_mask, other=0.0
            )

            # Store to linear pool
            pool_offset_lin = (
                pool_slot * max_codebook_size * vocab_size
                + codebook_idx * vocab_size
                + v_start
                + tl.arange(0, BLOCK_SIZE_V)
            )
            tl.atomic_add(linear_pool_ptr + pool_offset_lin, update_vals, mask=v_mask)


def update_hyper_weights_pooled(
    embedding_pool: torch.Tensor,
    linear_pool: torch.Tensor,
    updates: torch.Tensor,
    updates_indices: torch.Tensor,
    pool_indices: torch.Tensor,
    req_to_update_mapping: torch.Tensor,
):
    """
    Update hyper weights in the memory pool using Triton kernel.

    Args:
        embedding_pool: [pool_size, max_codebook_size, hidden_size]
        linear_pool: [pool_size, max_codebook_size, vocab_size]
        updates: [total_updates, hidden_size + vocab_size] - flattened updates
        updates_indices: [total_updates] - codebook indices for updates
        pool_indices: [num_requests] - pool slot for each request
        req_to_update_mapping: [num_requests, 2] - [start_idx, end_idx] for each request
    """
    num_requests = pool_indices.shape[0]
    max_codebook_size = embedding_pool.shape[1]
    hidden_size = embedding_pool.shape[2]
    vocab_size = linear_pool.shape[2]

    if num_requests == 0:
        return

    # Create start/end pointers
    req_to_update_start = req_to_update_mapping[:, 0].contiguous()
    req_to_update_end = req_to_update_mapping[:, 1].contiguous()

    # Launch kernel with one block per request
    grid = (num_requests,)

    hyper_weight_update_pooled_kernel[grid](
        embedding_pool,
        linear_pool,
        updates,
        updates_indices,
        pool_indices,
        req_to_update_start,
        req_to_update_end,
        max_codebook_size,
        hidden_size,
        vocab_size,
        num_requests,
        BLOCK_SIZE_H=64,
        BLOCK_SIZE_V=64,
    )
