"""
uam_airspace_context_od_v2.py — Fixed version of uam_airspace_context_od.py.

Changelog vs uam_airspace_context_od.py
-----------------------------------------
FIX-2  ContextModulatedDirectedGraphLearner: multi-head attention logits were
       averaged across heads *before* softmax (line 166 in original).
       Now softmax is applied independently per head on [B,T,H,N,N];
       the resulting attention weights are averaged afterwards.

FIX-3  prior_scale is now passed through F.softplus so it stays strictly
       positive and cannot invert the log-prior bias.
       Raw parameter is initialised to log(exp(1)-1) ≈ 0.5413 so the
       effective scale starts at 1.0 (same as the original).

FIX-4  valid_mask is applied to the full [B,T,H,N,N] logit tensor before
       the per-head softmax (consistent with FIX-2).  The original code
       masked a [B,T,N,N] tensor after the head-average, which was
       inconsistent with the intended per-head semantics.

FIX-5  ContextualODDemandEncoder and ContextAwareTemporalODDecoder both
       contained hard-coded 0.1 multipliers on FiLM scale and shift.
       This limited context modulation to ±10 % and made gradients very
       weak.  The multipliers are removed; tanh on scale still provides
       a (-1, +1) bound.

FIX-6  context_edge_scale initialised to 0.1 (was 0.05).  The prior_scale
       starts at 1.0 while context_edge_scale was 0.05 — a 20× gap that
       made the learned edge bias almost invisible early in training.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from uam_airspace_od import _as_float_tensor, _make_valid_mask, _support_to_prior


# ---------------------------------------------------------------------------
# Context encoder  (unchanged from original)
# ---------------------------------------------------------------------------

class WeatherCalendarContextEncoder(nn.Module):
    """Encode global weather and calendar signals separately from OD demand."""

    def __init__(self, context_dim: int, hidden_dim: int, num_steps: int, dropout: float):
        super().__init__()
        self.context_dim = int(context_dim)
        self.hidden_dim  = int(hidden_dim)
        self.input_norm  = nn.LayerNorm(context_dim)
        self.input_proj  = nn.Linear(context_dim, hidden_dim)
        self.time_embedding = nn.Parameter(torch.empty(num_steps, hidden_dim))
        self.temporal_gate  = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.time_embedding)
        nn.init.xavier_uniform_(self.temporal_gate.weight)
        nn.init.zeros_(self.temporal_gate.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: [B, T, E]
        h = self.input_proj(self.input_norm(context))
        h = h + self.time_embedding[None, :, :]
        residual = h
        mixed = self.temporal_gate(h.permute(0, 2, 1).contiguous())
        value, gate = mixed.chunk(2, dim=1)
        mixed = torch.tanh(value) * torch.sigmoid(gate)
        mixed = mixed.permute(0, 2, 1).contiguous()
        return self.norm(residual + self.dropout(mixed))


# ---------------------------------------------------------------------------
# FIX-5  ContextualODDemandEncoder — removed 0.1 FiLM multipliers
# ---------------------------------------------------------------------------

class ContextualODDemandEncoder(nn.Module):
    """Encode OD demand while modulating route states with weather-calendar context.

    FIX-5: The original hard-coded 0.1 multipliers on the FiLM scale/shift
    are removed.  tanh on scale still bounds it to (-1, +1).
    """

    def __init__(self, hidden_dim: int, num_nodes: int, num_steps: int, dropout: float):
        super().__init__()
        self.input_proj            = nn.Linear(1, hidden_dim)
        self.origin_embedding      = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.destination_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.edge_embedding        = nn.Parameter(torch.empty(num_nodes, num_nodes, hidden_dim))
        self.time_embedding        = nn.Parameter(torch.empty(num_steps, hidden_dim))
        self.context_film          = nn.Linear(hidden_dim, hidden_dim * 2)
        self.temporal_gate         = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.origin_embedding)
        nn.init.xavier_uniform_(self.destination_embedding)
        nn.init.xavier_uniform_(self.edge_embedding)
        nn.init.xavier_uniform_(self.time_embedding)
        nn.init.zeros_(self.context_film.weight)
        nn.init.zeros_(self.context_film.bias)
        nn.init.xavier_uniform_(self.temporal_gate.weight)
        nn.init.zeros_(self.temporal_gate.bias)

    def forward(self, demand: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        # demand: [B, O, D, 1, T],  context_state: [B, T, H]
        batch_size, num_origins, num_destinations, _, num_steps = demand.shape
        h = demand.permute(0, 1, 2, 4, 3)       # [B, O, D, T, 1]
        h = self.input_proj(h)                   # [B, O, D, T, H]
        h = h + self.origin_embedding[None, :, None, None, :]
        h = h + self.destination_embedding[None, None, :, None, :]
        h = h + self.edge_embedding[None, :, :, None, :]
        h = h + self.time_embedding[None, None, None, :, :]

        # FIX-5: removed 0.1 from FiLM scale and shift
        film  = self.context_film(context_state)[:, None, None, :, :]   # [B,1,1,T,2H]
        scale, shift = film.chunk(2, dim=-1)
        h = h * (1.0 + torch.tanh(scale)) + shift    # was: 0.1 * tanh, 0.1 * shift

        h = h.permute(0, 3, 1, 2, 4).contiguous()   # [B, T, O, D, H]
        residual = h
        route_series = h.permute(0, 2, 3, 4, 1).reshape(
            batch_size * num_origins * num_destinations, h.shape[-1], num_steps
        )
        gated = self.temporal_gate(route_series)
        value, gate = gated.chunk(2, dim=1)
        route_series = torch.tanh(value) * torch.sigmoid(gate)
        route_series = route_series.reshape(
            batch_size, num_origins, num_destinations, h.shape[-1], num_steps
        )
        route_series = route_series.permute(0, 4, 1, 2, 3).contiguous()
        return self.norm(residual + self.dropout(route_series))


# ---------------------------------------------------------------------------
# FIX-2/3/4/6  ContextModulatedDirectedGraphLearner
# ---------------------------------------------------------------------------

class ContextModulatedDirectedGraphLearnerV2(nn.Module):
    """Learn a directed graph with OD states and weather-calendar edge modulation.

    FIX-2  Softmax is applied per head on [B,T,H,N,N]; weights are averaged.
    FIX-3  prior_scale passed through F.softplus → always positive.
    FIX-4  valid_mask applied per-head before per-head softmax.
    FIX-6  context_edge_scale initialised to 0.1 (was 0.05).
    """

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
        use_context_graph: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads    = num_heads
        self.head_dim     = hidden_dim // num_heads
        self.use_context_graph = use_context_graph

        self.query_proj          = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj            = nn.Linear(hidden_dim, hidden_dim)
        self.context_query       = nn.Linear(hidden_dim, hidden_dim)
        self.context_key         = nn.Linear(hidden_dim, hidden_dim)
        self.context_edge_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.edge_context_embedding = nn.Parameter(
            torch.empty(num_nodes, num_nodes, hidden_dim)
        )

        # FIX-3: prior_scale via softplus; raw init so softplus(raw) ≈ 1.0
        _raw_init = math.log(math.exp(1.0) - 1.0)   # ≈ 0.5413
        self.prior_scale_raw = nn.Parameter(torch.tensor(_raw_init))

        # FIX-6: start at 0.1 instead of 0.05
        self.context_edge_scale = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(dropout)

        valid_mask = _make_valid_mask(num_nodes, corridor_mask, restricted_mask, allow_self)
        self.register_buffer("graph_prior", graph_prior.float())
        self.register_buffer("valid_mask", valid_mask)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.query_proj, self.key_proj,
            self.context_query, self.context_key, self.context_edge_proj,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.edge_context_embedding)

    @property
    def prior_scale(self) -> torch.Tensor:
        """Effective (always-positive) prior scale."""
        return F.softplus(self.prior_scale_raw)   # FIX-3

    def forward(self, h: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        # h: [B, T, O, D, H],  context_state: [B, T, H]
        batch_size, num_steps, num_nodes, _, hidden_dim = h.shape

        origin_state      = h.mean(dim=3)   # [B, T, N, H]
        destination_state = h.mean(dim=2)   # [B, T, N, H]
        if self.use_context_graph:
            origin_state      = origin_state      + self.context_query(context_state)[:, :, None, :]
            destination_state = destination_state + self.context_key(context_state)[:, :, None, :]

        query = self.query_proj(origin_state).view(
            batch_size, num_steps, num_nodes, self.num_heads, self.head_dim
        )
        key = self.key_proj(destination_state).view(
            batch_size, num_steps, num_nodes, self.num_heads, self.head_dim
        )

        # logits: [B, T, H, N, N]
        logits = torch.einsum("btihd,btjhd->bthij", query, key) / math.sqrt(self.head_dim)

        # FIX-3: add prior bias using softplus-constrained scale
        prior_log = torch.log(self.graph_prior.clamp_min(1e-6))          # [N, N]
        logits = logits + prior_log[None, None, None, :, :] * self.prior_scale

        # Add context edge bias (broadcast over heads)
        if self.use_context_graph:
            edge_ctx  = self.context_edge_proj(context_state)             # [B, T, H]
            edge_bias = torch.einsum(
                "bth,ijh->btij", edge_ctx, self.edge_context_embedding
            )                                                              # [B, T, N, N]
            logits = logits + self.context_edge_scale * edge_bias.unsqueeze(2) / math.sqrt(hidden_dim)

        # FIX-4: mask on [B,T,H,N,N] before per-head softmax
        logits = logits.masked_fill(
            ~self.valid_mask[None, None, None, :, :], -1e9
        )

        # FIX-2: softmax per head → average attention weights
        attn_per_head = F.softmax(logits, dim=-1)    # [B, T, H, N, N]
        return attn_per_head.mean(dim=2)             # [B, T, N, N]


# ---------------------------------------------------------------------------
# Message passing block  (graph learner swapped to V2)
# ---------------------------------------------------------------------------

class ContextualAirspaceODMessagePassingV2(nn.Module):
    """Propagate OD messages with context-conditioned route gates.

    Identical to the original ContextualAirspaceODMessagePassing except it
    uses ContextModulatedDirectedGraphLearnerV2 (FIX-2/3/4/6).
    """

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
        use_context_graph: bool = True,
        use_context_message: bool = True,
    ):
        super().__init__()
        self.use_context_message = use_context_message
        self.graph_learner = ContextModulatedDirectedGraphLearnerV2(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_nodes=num_nodes,
            graph_prior=graph_prior,
            corridor_mask=corridor_mask,
            restricted_mask=restricted_mask,
            allow_self=allow_self,
            dropout=dropout,
            use_context_graph=use_context_graph,
        )
        route_dim = hidden_dim * (4 if use_context_message else 3)
        self.message_proj = nn.Sequential(
            nn.Linear(route_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_proj     = nn.Linear(route_dim, hidden_dim)
        self.temporal_mixer = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.norm_message  = nn.LayerNorm(hidden_dim)
        self.norm_temporal = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, h: torch.Tensor, context_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [B, T, O, D, H]
        batch_size, num_steps, num_origins, num_destinations, hidden_dim = h.shape
        dynamic_adj = self.graph_learner(h, context_state)   # [B, T, N, N]

        origin_context      = torch.einsum("btij,btijd->btid", dynamic_adj, h)
        destination_context = torch.einsum("btij,btijd->btjd", dynamic_adj, h)
        origin_context      = origin_context.unsqueeze(3).expand(-1, -1, -1, num_destinations, -1)
        destination_context = destination_context.unsqueeze(2).expand(-1, -1, num_origins, -1, -1)

        pieces = [h, origin_context, destination_context]
        if self.use_context_message:
            route_ctx = context_state[:, :, None, None, :].expand(
                -1, -1, num_origins, num_destinations, -1
            )
            pieces.append(route_ctx)
        msg_input = torch.cat(pieces, dim=-1)
        gate    = torch.sigmoid(self.gate_proj(msg_input))
        message = self.message_proj(msg_input)
        h = self.norm_message(h + self.dropout(gate * message))

        route_series = h.permute(0, 2, 3, 4, 1).reshape(
            batch_size * num_origins * num_destinations, hidden_dim, num_steps
        )
        mixed = self.temporal_mixer(route_series)
        value, mix_gate = mixed.chunk(2, dim=1)
        mixed = torch.tanh(value) * torch.sigmoid(mix_gate)
        mixed = mixed.reshape(
            batch_size, num_origins, num_destinations, hidden_dim, num_steps
        )
        mixed = mixed.permute(0, 4, 1, 2, 3).contiguous()
        h = self.norm_temporal(h + self.dropout(mixed))
        return h, dynamic_adj


# ---------------------------------------------------------------------------
# FIX-5  ContextAwareTemporalODDecoder — removed 0.1 FiLM multipliers
# ---------------------------------------------------------------------------

class ContextAwareTemporalODDecoderV2(nn.Module):
    """Attention-pool historical OD states with context-conditioned decoding.

    FIX-5: Removed the 0.1 multipliers on FiLM scale/shift (line 296 in original).
    """

    def __init__(
        self,
        hidden_dim: int,
        out_steps: int,
        dropout: float,
        nonnegative_output: bool = True,
        use_context_decoder: bool = True,
    ):
        super().__init__()
        self.use_context_decoder = use_context_decoder
        self.score_proj    = nn.Linear(hidden_dim, 1)
        self.context_score = nn.Linear(hidden_dim, hidden_dim)
        self.context_film  = nn.Linear(hidden_dim, hidden_dim * 2)
        self.output_proj   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_steps),
        )
        self.nonnegative_output = nonnegative_output
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.context_score.weight)
        nn.init.zeros_(self.context_score.bias)
        nn.init.zeros_(self.context_film.weight)
        nn.init.zeros_(self.context_film.bias)

    def forward(self, h: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        # h: [B, T, O, D, H],  context_state: [B, T, H]
        score_input = h
        if self.use_context_decoder:
            score_input = score_input + self.context_score(context_state)[:, :, None, None, :]
        time_score  = self.score_proj(score_input).squeeze(-1)   # [B, T, O, D]
        time_weight = F.softmax(time_score, dim=1).unsqueeze(-1) # [B, T, O, D, 1]
        pooled = (h * time_weight).sum(dim=1)                    # [B, O, D, H]

        if self.use_context_decoder:
            pooled_context = (context_state[:, :, None, None, :] * time_weight).sum(dim=1)
            scale, shift = self.context_film(pooled_context).chunk(2, dim=-1)
            # FIX-5: removed 0.1 multipliers
            pooled = pooled * (1.0 + torch.tanh(scale)) + shift

        y = self.output_proj(pooled)
        if self.nonnegative_output:
            y = F.softplus(y)
        return y


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ContextAwareUAMNetV2(nn.Module):
    """UAM OD demand model with separate weather-calendar context modulation — v2.

    All fixes (FIX-2 through FIX-6) are applied.  Architecture is identical
    to ContextAwareUAMNet; only the sub-modules are swapped.

    Input shape:  [batch, origin, destination, channel, history_step]
    Output shape: [batch, origin, destination, out_step]
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
        use_context_graph: bool = True,
        use_context_message: bool = True,
        use_context_decoder: bool = True,
        use_context_film: bool = True,
    ):
        super().__init__()
        num_blocks, num_heads, head_dim = [int(v) for v in arg_set]
        num_steps, num_nodes, in_channels = [int(v) for v in x_size]
        if in_channels < 1:
            raise ValueError("in_channels must be at least 1")

        hidden_dim  = num_heads * head_dim
        context_dim = max(in_channels - 1, 1)
        dropout     = min(max(float(bn_decay), 0.0), 0.5)

        graph_support = _as_float_tensor(graph_support)
        graph_prior   = _support_to_prior(graph_support, num_nodes)
        if isinstance(graph_support, torch.Tensor):
            graph_prior = graph_prior.to(graph_support.device)

        self.in_channels      = in_channels
        self.hidden_dim       = hidden_dim
        self.use_context_film = use_context_film and in_channels > 1

        self.context_encoder = WeatherCalendarContextEncoder(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            dropout=dropout,
        )
        self.demand_encoder = ContextualODDemandEncoder(   # FIX-5 applied
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_steps=num_steps,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                ContextualAirspaceODMessagePassingV2(   # FIX-2/3/4/6 applied
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_nodes=num_nodes,
                    graph_prior=graph_prior,
                    corridor_mask=corridor_mask,
                    restricted_mask=restricted_mask,
                    allow_self=allow_self,
                    dropout=dropout,
                    use_context_graph=use_context_graph and in_channels > 1,
                    use_context_message=use_context_message and in_channels > 1,
                )
                for _ in range(num_blocks)
            ]
        )
        self.decoder = ContextAwareTemporalODDecoderV2(   # FIX-5 applied
            hidden_dim=hidden_dim,
            out_steps=out_steps,
            dropout=dropout,
            nonnegative_output=nonnegative_output,
            use_context_decoder=use_context_decoder and in_channels > 1,
        )
        self.latest_dynamic_adj = None

    def extract_context(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels <= 1:
            b, _, _, _, t = x.shape
            return x.new_zeros(b, t, self.hidden_dim)
        context = x[:, :, :, 1:self.in_channels, :].mean(dim=(1, 2))
        return self.context_encoder(context.permute(0, 2, 1).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        demand        = x[:, :, :, 0:1, :]
        context_state = self.extract_context(x)
        ctx_for_enc   = context_state if self.use_context_film else torch.zeros_like(context_state)

        h = self.demand_encoder(demand, ctx_for_enc)
        dynamic_adj = None
        for block in self.blocks:
            h, dynamic_adj = block(h, context_state)
        self.latest_dynamic_adj = None if dynamic_adj is None else dynamic_adj.detach()
        return self.decoder(h, context_state)


class ASTT(ContextAwareUAMNetV2):
    """Drop-in alias so training scripts can call ASTT(graph, arg_set, x_size, bn_decay)."""
    pass


if __name__ == "__main__":
    support = torch.eye(20).repeat(3, 1, 1)
    model   = ContextAwareUAMNetV2(support, [1, 8, 8], [12, 20, 12], 0.1)
    sample  = torch.randn(4, 20, 20, 12, 12)
    output  = model(sample)
    print("output shape:", tuple(output.shape))   # expect (4, 20, 20, 1)
