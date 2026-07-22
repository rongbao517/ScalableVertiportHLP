import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


def _support_to_prior(support: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Convert a graph support or Chebyshev basis into a row-normalized prior."""
    support = _as_float_tensor(support)
    if support.ndim == 3:
        prior = support.abs().sum(dim=0)
    elif support.ndim == 2:
        prior = support.abs()
    else:
        raise ValueError("graph support must have shape [N, N] or [K, N, N]")

    if prior.shape[-2:] != (num_nodes, num_nodes):
        raise ValueError(f"graph support shape {tuple(prior.shape)} does not match num_nodes={num_nodes}")

    prior = prior.clone()
    prior.fill_diagonal_(0.0)

    row_sum = prior.sum(dim=-1, keepdim=True)
    empty_rows = row_sum.squeeze(-1) <= 1e-8
    if empty_rows.any():
        fallback = torch.ones_like(prior)
        fallback.fill_diagonal_(0.0)
        prior[empty_rows] = fallback[empty_rows]
        row_sum = prior.sum(dim=-1, keepdim=True)

    return prior / row_sum.clamp_min(1e-8)


def _make_valid_mask(
    num_nodes: int,
    corridor_mask: Optional[torch.Tensor] = None,
    restricted_mask: Optional[torch.Tensor] = None,
    allow_self: bool = False,
) -> torch.Tensor:
    valid = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
    if not allow_self:
        valid.fill_diagonal_(False)

    if corridor_mask is not None:
        valid = valid & (_as_float_tensor(corridor_mask).cpu() > 0)
    if restricted_mask is not None:
        valid = valid & (_as_float_tensor(restricted_mask).cpu() > 0)

    row_empty = ~valid.any(dim=-1)
    if row_empty.any():
        fallback = torch.ones_like(valid)
        if not allow_self:
            fallback.fill_diagonal_(False)
        valid[row_empty] = fallback[row_empty]

    return valid


class ODPairTemporalEncoder(nn.Module):
    """Encode each directed OD pair without collapsing the OD matrix into node features."""

    def __init__(self, in_channels: int, hidden_dim: int, num_nodes: int, num_steps: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.origin_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.destination_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.edge_embedding = nn.Parameter(torch.empty(num_nodes, num_nodes, hidden_dim))
        self.time_embedding = nn.Parameter(torch.empty(num_steps, hidden_dim))
        self.temporal_gate = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.origin_embedding)
        nn.init.xavier_uniform_(self.destination_embedding)
        nn.init.xavier_uniform_(self.edge_embedding)
        nn.init.xavier_uniform_(self.time_embedding)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.temporal_gate.weight)
        nn.init.zeros_(self.temporal_gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, O, D, C, T]
        if x.ndim != 5:
            raise ValueError(f"expected input shape [B, O, D, C, T], got {tuple(x.shape)}")

        batch_size, num_origins, num_destinations, _, num_steps = x.shape
        h = x.permute(0, 1, 2, 4, 3)
        h = self.input_proj(h)

        h = h + self.origin_embedding[None, :, None, None, :]
        h = h + self.destination_embedding[None, None, :, None, :]
        h = h + self.edge_embedding[None, :, :, None, :]
        h = h + self.time_embedding[None, None, None, :, :]
        h = h.permute(0, 3, 1, 2, 4).contiguous()

        residual = h
        route_series = h.permute(0, 2, 3, 4, 1).reshape(
            batch_size * num_origins * num_destinations, h.shape[-1], num_steps
        )
        gated = self.temporal_gate(route_series)
        value, gate = gated.chunk(2, dim=1)
        route_series = torch.tanh(value) * torch.sigmoid(gate)
        route_series = route_series.reshape(batch_size, num_origins, num_destinations, h.shape[-1], num_steps)
        route_series = route_series.permute(0, 4, 1, 2, 3).contiguous()

        return self.norm(residual + self.dropout(route_series))


class DynamicDirectedGraphLearner(nn.Module):
    """Learn a time-varying directed graph under static UAM airspace constraints."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_nodes: int,
        graph_prior: torch.Tensor,
        corridor_mask: Optional[torch.Tensor],
        restricted_mask: Optional[torch.Tensor],
        allow_self: bool,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

        valid_mask = _make_valid_mask(num_nodes, corridor_mask, restricted_mask, allow_self)
        self.register_buffer("graph_prior", graph_prior.float())
        self.register_buffer("valid_mask", valid_mask)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, O, D, H]
        batch_size, num_steps, num_nodes, _, hidden_dim = h.shape

        origin_state = h.mean(dim=3)
        destination_state = h.mean(dim=2)

        query = self.query_proj(origin_state).view(
            batch_size, num_steps, num_nodes, self.num_heads, self.head_dim
        )
        key = self.key_proj(destination_state).view(
            batch_size, num_steps, num_nodes, self.num_heads, self.head_dim
        )

        logits = torch.einsum("btihd,btjhd->bthij", query, key) / math.sqrt(self.head_dim)
        logits = logits.mean(dim=2)

        prior_bias = torch.log(self.graph_prior.clamp_min(1e-6)) * self.prior_scale
        logits = logits + prior_bias[None, None, :, :]
        logits = logits.masked_fill(~self.valid_mask[None, None, :, :], -1e9)
        return F.softmax(logits, dim=-1)


class AirspaceODMessagePassing(nn.Module):
    """Propagate dependencies among OD routes sharing origins, destinations, and airspace constraints."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_nodes: int,
        graph_prior: torch.Tensor,
        corridor_mask: Optional[torch.Tensor],
        restricted_mask: Optional[torch.Tensor],
        allow_self: bool,
        dropout: float,
    ):
        super().__init__()
        self.graph_learner = DynamicDirectedGraphLearner(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_nodes=num_nodes,
            graph_prior=graph_prior,
            corridor_mask=corridor_mask,
            restricted_mask=restricted_mask,
            allow_self=allow_self,
            dropout=dropout,
        )
        self.message_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.temporal_mixer = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.norm_message = nn.LayerNorm(hidden_dim)
        self.norm_temporal = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [B, T, O, D, H]
        batch_size, num_steps, num_origins, num_destinations, hidden_dim = h.shape
        dynamic_adj = self.graph_learner(h)

        origin_context = torch.einsum("btij,btijd->btid", dynamic_adj, h)
        destination_context = torch.einsum("btij,btijd->btjd", dynamic_adj, h)
        origin_context = origin_context.unsqueeze(3).expand(-1, -1, -1, num_destinations, -1)
        destination_context = destination_context.unsqueeze(2).expand(-1, -1, num_origins, -1, -1)

        route_context = torch.cat([h, origin_context, destination_context], dim=-1)
        gate = torch.sigmoid(self.gate_proj(route_context))
        message = self.message_proj(route_context)
        h = self.norm_message(h + self.dropout(gate * message))

        route_series = h.permute(0, 2, 3, 4, 1).reshape(
            batch_size * num_origins * num_destinations, hidden_dim, num_steps
        )
        mixed = self.temporal_mixer(route_series)
        value, mix_gate = mixed.chunk(2, dim=1)
        mixed = torch.tanh(value) * torch.sigmoid(mix_gate)
        mixed = mixed.reshape(batch_size, num_origins, num_destinations, hidden_dim, num_steps)
        mixed = mixed.permute(0, 4, 1, 2, 3).contiguous()
        h = self.norm_temporal(h + self.dropout(mixed))

        return h, dynamic_adj


class TemporalODDecoder(nn.Module):
    """Attention-pool historical OD states and predict future OD demand."""

    def __init__(self, hidden_dim: int, out_steps: int, dropout: float, nonnegative_output: bool = True):
        super().__init__()
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_steps),
        )
        self.nonnegative_output = nonnegative_output

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, O, D, H]
        time_score = self.score_proj(h).squeeze(-1)
        time_weight = F.softmax(time_score, dim=1).unsqueeze(-1)
        pooled = (h * time_weight).sum(dim=1)
        y = self.output_proj(pooled)
        if self.nonnegative_output:
            y = F.softplus(y)
        return y


class AirspaceConstrainedUAMNet(nn.Module):
    """UAM-specific OD demand model with airspace-aware dynamic directed graph learning.

    Input shape:
        [batch, origin, destination, channel, history_step]

    Output shape:
        [batch, origin, destination, out_step]

    Main design choices:
        1. Preserve OD-pair states instead of collapsing OD pairs into node states.
        2. Learn a time-varying directed graph from origin and destination route states.
        3. Constrain graph learning with optional corridor and restricted-airspace masks.
        4. Propagate same-origin and same-destination route interactions.
    """

    def __init__(
        self,
        graph_support: torch.Tensor,
        arg_set: Sequence[int],
        x_size: Sequence[int],
        bn_decay: float = 0.2,
        out_steps: int = 1,
        corridor_mask: Optional[torch.Tensor] = None,
        restricted_mask: Optional[torch.Tensor] = None,
        allow_self: bool = False,
        nonnegative_output: bool = True,
    ):
        super().__init__()
        if len(arg_set) != 3:
            raise ValueError("arg_set must be [num_blocks, num_heads, head_dim]")
        if len(x_size) != 3:
            raise ValueError("x_size must be [num_steps, num_nodes, in_channels]")

        num_blocks, num_heads, head_dim = [int(v) for v in arg_set]
        num_steps, num_nodes, in_channels = [int(v) for v in x_size]
        hidden_dim = num_heads * head_dim
        dropout = min(max(float(bn_decay), 0.0), 0.5)

        graph_support = _as_float_tensor(graph_support)
        graph_prior = _support_to_prior(graph_support, num_nodes)
        if isinstance(graph_support, torch.Tensor):
            graph_prior = graph_prior.to(graph_support.device)

        self.encoder = ODPairTemporalEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_steps=num_steps,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                AirspaceODMessagePassing(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_nodes=num_nodes,
                    graph_prior=graph_prior,
                    corridor_mask=corridor_mask,
                    restricted_mask=restricted_mask,
                    allow_self=allow_self,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.decoder = TemporalODDecoder(
            hidden_dim=hidden_dim,
            out_steps=out_steps,
            dropout=dropout,
            nonnegative_output=nonnegative_output,
        )
        self.latest_dynamic_adj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        dynamic_adj = None
        for block in self.blocks:
            h, dynamic_adj = block(h)
        self.latest_dynamic_adj = None if dynamic_adj is None else dynamic_adj.detach()
        return self.decoder(h)


class ASTT(AirspaceConstrainedUAMNet):
    """Drop-in alias for training scripts that instantiate ASTT(graph, arg_set, x_size, bn_decay)."""

    pass


if __name__ == "__main__":
    support = torch.eye(15).repeat(3, 1, 1)
    model = AirspaceConstrainedUAMNet(support, [2, 8, 8], [12, 15, 12], 0.1)
    sample = torch.randn(4, 15, 15, 12, 12)
    output = model(sample)
    print("output shape:", tuple(output.shape))
