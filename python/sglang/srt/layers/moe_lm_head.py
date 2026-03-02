"""Mixture-of-Experts based Language Model Head.

Partitions the vocabulary across num_experts sub-vocabularies. A learned router
selects top_k experts per token, and only those experts' sub-vocabulary logits
are computed. Unselected positions are filled with -inf.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


class MoELMHead(nn.Module):
    """MoE-based language model head.

    Each expert owns a contiguous slice of the vocabulary. The router selects
    top_k experts per token; only selected experts compute logits via a single
    GEMM. Unselected vocabulary positions are filled with -inf.

    Args:
        vocab_size: Total vocabulary size.
        hidden_size: Model hidden dimension.
        num_experts: Number of vocabulary partition experts.
        top_k: Number of experts activated per token.
        num_fused_shared_experts: Always-active shared partition experts (reserved for future use).
        scoring_func: Routing scoring function ("softmax" or "sigmoid").
        params_dtype: Parameter data type.
        prefix: Weight name prefix for loading.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        num_fused_shared_experts: int = 0,
        scoring_func: str = "softmax",
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_fused_shared_experts = num_fused_shared_experts
        self.scoring_func = scoring_func
        self.params_dtype = params_dtype
        self.prefix = prefix

        # Each expert owns a contiguous chunk of the vocabulary.
        # Pad up so all experts have equal sub-vocab size.
        self.expert_vocab_size = math.ceil(vocab_size / num_experts)

        # Expert projection weights: [num_experts, expert_vocab_size, hidden_size]
        self.expert_weight = Parameter(
            torch.empty(
                num_experts,
                self.expert_vocab_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            self.expert_weight,
            {"weight_loader": self.weight_loader},
        )

        # Router gate: [num_experts, hidden_size]
        # Initialized with kaiming_uniform since standard checkpoints won't have
        # router weights for the lm_head.
        self.router_weight = Parameter(
            torch.empty(num_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        torch.nn.init.kaiming_uniform_(self.router_weight, a=math.sqrt(5))
        set_weight_attrs(
            self.router_weight,
            {"weight_loader": self.weight_loader},
        )

        # Precompute per-expert valid lengths (last expert may have fewer tokens)
        expert_valid_lens = []
        for e in range(num_experts):
            start = e * self.expert_vocab_size
            end = min(start + self.expert_vocab_size, vocab_size)
            expert_valid_lens.append(end - start)
        self.register_buffer(
            "expert_valid_lens",
            torch.tensor(expert_valid_lens, dtype=torch.long),
            persistent=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute MoE lm_head logits.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            logits: [num_tokens, vocab_size] with -inf for unselected positions.
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device

        # Step 1: Compute router logits and select top-k experts
        router_logits = torch.matmul(
            hidden_states.to(self.router_weight.dtype),
            self.router_weight.T,
        )  # [num_tokens, num_experts]

        if self.scoring_func == "softmax":
            routing_weights = torch.softmax(router_logits, dim=-1)
        elif self.scoring_func == "sigmoid":
            routing_weights = torch.sigmoid(router_logits)
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_func}")

        # topk_weights are kept for potential auxiliary loss, NOT for scaling logits.
        # Each expert produces logits for disjoint vocab positions, so scaling
        # would distort the probability distribution.
        _topk_weights, topk_ids = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # [num_tokens, top_k]

        # Step 2: Initialize output with -inf
        logits = torch.full(
            (num_tokens, self.vocab_size),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )

        # Step 3: Flatten expert assignments for efficient grouping
        # flat_expert_ids: [num_tokens * top_k] — which expert each (token, k) slot uses
        flat_expert_ids = topk_ids.reshape(-1)
        # flat_token_indices: [num_tokens * top_k] — which token each slot belongs to
        flat_token_indices = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )

        # Step 4: For each expert, gather assigned tokens, compute GEMM, scatter
        for expert_id in range(self.num_experts):
            mask = flat_expert_ids == expert_id
            if not mask.any():
                continue

            token_indices = flat_token_indices[mask]  # [n]
            selected_hidden = hidden_states[token_indices]  # [n, hidden_size]

            # Single GEMM: [n, hidden_size] x [expert_vocab_size, hidden_size].T
            expert_logits = torch.matmul(
                selected_hidden.to(self.expert_weight.dtype),
                self.expert_weight[expert_id].T,
            ).float()  # [n, expert_vocab_size]

            # Scatter into full vocab at correct positions
            valid_len = self.expert_valid_lens[expert_id].item()
            vocab_start = expert_id * self.expert_vocab_size

            logits[token_indices, vocab_start : vocab_start + valid_len] = (
                expert_logits[:, :valid_len]
            )

        return logits

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: Optional[str] = None,
        expert_id: Optional[int] = None,
    ) -> None:
        """Load weights from checkpoint.

        Supports two checkpoint formats:
        1. Per-expert weights:
           - lm_head.router.weight -> router_weight [num_experts, hidden_size]
           - lm_head.experts.{e}.weight -> expert_weight[e] [expert_vocab_size, hidden_size]
        2. Monolithic lm_head weight: [vocab_size, hidden_size]
           Partitioned across experts automatically.
        """
        if param is self.router_weight:
            if expert_id is not None:
                param.data[expert_id].copy_(loaded_weight)
            else:
                param.data.copy_(loaded_weight)
            return

        if param is self.expert_weight:
            if expert_id is not None:
                # Per-expert loading
                valid_len = self.expert_valid_lens[expert_id].item()
                param.data[expert_id, :valid_len, :].copy_(
                    loaded_weight[:valid_len, :]
                )
                # Zero-fill padding rows
                if valid_len < self.expert_vocab_size:
                    param.data[expert_id, valid_len:, :].zero_()
            else:
                # Monolithic lm_head weight: partition across experts
                for e in range(self.num_experts):
                    start = e * self.expert_vocab_size
                    end = min(start + self.expert_vocab_size, loaded_weight.shape[0])
                    length = end - start
                    param.data[e, :length, :].copy_(loaded_weight[start:end, :])
                    # Zero-fill padding rows
                    if length < self.expert_vocab_size:
                        param.data[e, length:, :].zero_()
            return

        # Fallback: direct copy
        param.data.copy_(loaded_weight)
