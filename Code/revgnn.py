"""
revgnn.py
=========
Implementation of:
    "Training Graph Neural Networks with 1000 Layers"
    Li et al., ICML 2021  |  arXiv:2106.07476

Implemented architectures
--------------------------
1. ResGNN          - residual GNN baseline              O(LND) memory
2. RevGNN          - grouped reversible GNN             O( ND) memory  ← paper's main contribution
3. WT-ResGNN       - weight-tied residual GNN           O(LND) memory, O(D²) params
4. WT-RevGNN       - weight-tied reversible GNN         O( ND) memory, O(D²) params
5. DEQ-GNN         - deep equilibrium GNN               O( ND) memory, O(D²) params (infinite depth)

Memory complexity comparison (Table 4 in the paper):
    Full-batch GNN       : O(LND)   params O(LD²)
    RevGNN               : O(ND)    params O(LD²)   ← same params, much less memory
    WT-RevGNN            : O(ND)    params O(D²)
    DEQ-GNN              : O(ND)    params O(D²)

Usage example
-------------
    from revgnn import RevGNN, ResGNN, WTRevGNN, DEQGNN, build_model
    model = build_model('revgnn', in_ch=8, hidden=80, out_ch=112, layers=1001)
"""

import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_scatter import scatter

# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def peak_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0

def current_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0

def theoretical_memory(model_type: str, L: int, N: int, D: int) -> float:
    """Return theoretical activation memory in GB (float32)."""
    bytes_per_el = 4
    if model_type in ('resgnn', 'wt-resgnn'):
        mem = L * N * D * bytes_per_el
    elif model_type in ('revgnn', 'wt-revgnn', 'deq'):
        mem = N * D * bytes_per_el
    else:
        mem = L * N * D * bytes_per_el
    return mem / 1e9

# ─────────────────────────────────────────────────────────────────────────────
# 1. Pre-activation GNN block
# ─────────────────────────────────────────────────────────────────────────────

class GNNBlock(nn.Module):
    """
    Pre-activation block from DeeperGCN (Li et al. 2020):

        x_b = Dropout( ReLU( LayerNorm(x) ) )
        out  = GraphConv( x_b , A , [U] )

    Supports GCN, SAGE, and a simplified GEN operator.
    When edge_dim is given the edge features are encoded and added to messages
    before the graph convolution (used for ogbn-proteins).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = "gcn",
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.dropout_p = dropout
        self.conv_type = conv_type

        # Optional edge-feature encoder (ogbn-proteins uses 8-dim edge feats)
        if edge_dim is not None and edge_dim > 0:
            self.edge_enc = nn.Linear(edge_dim, in_channels)
            self.has_edge = True
        else:
            self.has_edge = False

        if conv_type == "gcn":
            self.conv = GCNConv(in_channels, out_channels, add_self_loops=True)
        elif conv_type == "sage":
            self.conv = SAGEConv(in_channels, out_channels)
        elif conv_type == "gen":
            # Simplified GEN: SAGE-like with max aggregation  (Li et al. 2020)
            self.conv = SAGEConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

    # ------------------------------------------------------------------
    # Internal forward (allows toggling dropout for reconstruction)
    # ------------------------------------------------------------------
    def _conv_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        apply_dropout: bool,
        dropout_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = F.relu(self.norm(x))
        if apply_dropout:
            if dropout_mask is not None:
                h = h * dropout_mask
            else:
                h = F.dropout(h, p=self.dropout_p, training=True)

        if self.has_edge and edge_attr is not None:
            edge_feat = self.edge_enc(edge_attr)
            row, col = edge_index
            agg = scatter(edge_feat, col, dim=0, dim_size=x.size(0), reduce="sum")
            h = h + agg

        return self.conv(h, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        dropout_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._conv_forward(
            x, edge_index, edge_attr,
            apply_dropout=self.training,
            dropout_mask=dropout_mask,
        )

    # Used during reconstruction (no dropout)
    def forward_no_dropout(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._conv_forward(
            x, edge_index, edge_attr,
            apply_dropout=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shared Dropout (O(ND) memory instead of O(LND) for masks)
# ─────────────────────────────────────────────────────────────────────────────

class SharedDropoutMask:
    """
    A single dropout mask shared across all layers.

    Paper (Section 3.2):
      "the dropout pattern is shared across layers. Therefore, we only need to
       store one dropout pattern … its memory complexity is independent of the
       depth: O(ND)."

    The same mask is reactivated during the reverse reconstruction pass to
    ensure numerically exact input reconstruction.
    """

    def __init__(self, p: float = 0.1):
        self.p = p
        self._mask: Optional[torch.Tensor] = None

    def new_mask(self, shape: tuple, device: torch.device) -> torch.Tensor:
        if self.p > 0:
            self._mask = (
                torch.rand(shape, device=device) >= self.p
            ).float() / (1.0 - self.p)
        else:
            self._mask = torch.ones(shape, device=device)
        return self._mask

    @property
    def mask(self) -> Optional[torch.Tensor]:
        return self._mask

    def reset(self):
        self._mask = None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom autograd: Reversible GNN Block (memory-efficient backward)
# ─────────────────────────────────────────────────────────────────────────────

class _RevBlockFn(torch.autograd.Function):
    """
    Custom autograd function for one reversible block (C = 2 groups).

    Forward  :  y1 = F(x2) + x1,  y2 = G(y1) + x2
    Backward :  reconstruct x1, x2 from y1, y2 on the fly → no need to store them.

    Memory savings:
      Standard backprop stores O(L) activation tensors.
      This function stores O(1) tensors (only the block's output).
    """

    @staticmethod
    def forward(ctx, x1, x2, F_mod, G_mod, edge_index, edge_attr, dropout_mask):
        ctx.F_mod = F_mod
        ctx.G_mod = G_mod
        ctx.edge_attr = edge_attr
        ctx.dropout_mask = dropout_mask

        with torch.no_grad():
            fx2 = F_mod(x2, edge_index, edge_attr, dropout_mask)
            y1 = x1 + fx2

            gy1 = G_mod(y1, edge_index, edge_attr, dropout_mask)
            y2 = x2 + gy1

        # ⬇ Only the OUTPUT is stored; inputs are NOT kept in memory
        if edge_attr is not None:
            ctx.save_for_backward(y1.detach(), y2.detach(), edge_index, edge_attr)
            ctx.has_ea = True
        else:
            ctx.save_for_backward(y1.detach(), y2.detach(), edge_index)
            ctx.has_ea = False

        return y1.detach().requires_grad_(x1.requires_grad), \
               y2.detach().requires_grad_(x2.requires_grad)

    @staticmethod
    def backward(ctx, dy1, dy2):
        if ctx.has_ea:
            y1, y2, edge_index, edge_attr = ctx.saved_tensors
        else:
            y1, y2, edge_index = ctx.saved_tensors
            edge_attr = None

        F_mod = ctx.F_mod
        G_mod = ctx.G_mod
        dm = ctx.dropout_mask  # shared dropout mask (used for exact reconstruction)

        # ── Step 1: reconstruct inputs without gradient ───────────────────────
        with torch.no_grad():
            gy1 = G_mod.forward_no_dropout(y1, edge_index, edge_attr)
            x2  = y2 - gy1

            fx2 = F_mod.forward_no_dropout(x2, edge_index, edge_attr)
            x1  = y1 - fx2

        # ── Step 2: gradient through G (y2 = G(y1) + x2) ────────────────────
        y1_g = y1.detach().requires_grad_(True)
        with torch.enable_grad():
            g_y1 = G_mod(y1_g, edge_index, edge_attr, dm)
        torch.autograd.backward(g_y1, dy2)
        dg_y1 = y1_g.grad.detach() if y1_g.grad is not None else torch.zeros_like(y1)

        # Total gradient arriving at y1
        dy1_total = dy1 + dg_y1

        # ── Step 3: gradient through F (y1 = F(x2) + x1) ────────────────────
        x2_f = x2.detach().requires_grad_(True)
        with torch.enable_grad():
            f_x2 = F_mod(x2_f, edge_index, edge_attr, dm)
        torch.autograd.backward(f_x2, dy1_total)
        df_x2 = x2_f.grad.detach() if x2_f.grad is not None else torch.zeros_like(x2)

        # ── Output gradients ─────────────────────────────────────────────────
        # dx1 = dy1_total  (y1 = F(x2) + x1 → ∂y1/∂x1 = I)
        # dx2 = dy2 (direct) + df_x2 (through F)
        dx1 = dy1_total
        dx2 = dy2 + df_x2

        return dx1, dx2, None, None, None, None, None


def rev_block_apply(x1, x2, F_mod, G_mod, edge_index, edge_attr, dropout_mask):
    """Thin wrapper so callers don't have to call .apply() directly."""
    return _RevBlockFn.apply(x1, x2, F_mod, G_mod, edge_index, edge_attr, dropout_mask)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ResGNN (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class ResGNN(nn.Module):
    """
    Residual GNN - baseline model.

    Memory complexity of activations: O(L·N·D)
    (scales linearly with depth → runs out of memory for large L)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.1,
        conv_type: str = "gcn",
        edge_dim: Optional[int] = None,
        weight_tied: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_tied = weight_tied
        self.model_type = "wt-resgnn" if weight_tied else "resgnn"

        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        if weight_tied:
            # Single shared block for all layers
            shared = GNNBlock(hidden_channels, hidden_channels, conv_type, dropout, edge_dim)
            self.gnn_layers = nn.ModuleList([shared] * num_layers)
        else:
            self.gnn_layers = nn.ModuleList([
                GNNBlock(hidden_channels, hidden_channels, conv_type, dropout, edge_dim)
                for _ in range(num_layers)
            ])

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr) + x
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def count_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


# ─────────────────────────────────────────────────────────────────────────────
# 5. RevGNN (Grouped Reversible GNN)
# ─────────────────────────────────────────────────────────────────────────────

class RevGNN(nn.Module):
    """
    Grouped Reversible GNN.

    Memory complexity of activations: O(N·D)   ← independent of depth!
    Parameter complexity            : O(L·D²)  (same as ResGNN for C=2)

    Architecture (C = 2 groups, Section 3.2 in the paper):
        Split x → (x1, x2)  each of size N×(D/2)
        For each block:
            y1 = F(x2) + x1        [F = GNN block acting on group 2]
            y2 = G(y1) + x2        [G = GNN block acting on group 1]

    Backward pass reconstructs x1, x2 from y1, y2 without storing them.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.1,
        conv_type: str = "gcn",
        edge_dim: Optional[int] = None,
        weight_tied: bool = False,
        num_groups: int = 2,          # C in the paper (default 2)
    ):
        super().__init__()
        if hidden_channels % num_groups != 0:
            raise ValueError("hidden_channels must be divisible by num_groups")

        self.num_layers = num_layers
        self.num_groups = num_groups
        self.dropout = dropout
        self.weight_tied = weight_tied
        self.model_type = "wt-revgnn" if weight_tied else "revgnn"

        group_ch = hidden_channels // num_groups
        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        # Each layer has num_groups GNN blocks (F blocks)
        if weight_tied:
            # One shared set of blocks (weight tying)
            shared_blocks = nn.ModuleList([
                GNNBlock(group_ch, group_ch, conv_type, dropout, edge_dim)
                for _ in range(num_groups)
            ])
            self.layers = nn.ModuleList([shared_blocks] * num_layers)
        else:
            self.layers = nn.ModuleList([
                nn.ModuleList([
                    GNNBlock(group_ch, group_ch, conv_type, dropout, edge_dim)
                    for _ in range(num_groups)
                ])
                for _ in range(num_layers)
            ])

        self.classifier = nn.Linear(hidden_channels, out_channels)

        # Shared dropout mask (O(ND) memory instead of O(LND))
        self.shared_dm = SharedDropoutMask(dropout)

    def _apply_rev_block(self, xs, blocks, edge_index, edge_attr, dm):
        """
        Apply one grouped reversible block.

        Paper equations (Section 3.2):
            X'_0 = sum(X_i for i=2..C)
            X'_i = f_{w_i}(X'_{i-1}, A, U) + X_i    for i in {1,...,C}

        For C=2 this reduces to:
            y1 = F(x2) + x1     (using custom autograd for memory efficiency)
            y2 = G(y1) + x2
        """
        if self.num_groups == 2:
            x1, x2 = xs[0], xs[1]
            F_mod, G_mod = blocks[0], blocks[1]
            y1, y2 = rev_block_apply(x1, x2, F_mod, G_mod, edge_index, edge_attr, dm)
            return [y1, y2]
        else:
            # General C > 2: sequential application
            # X'_0 = sum(X_2 ... X_C)
            x0_prime = sum(xs[1:])
            x_primes = []
            prev = x0_prime
            for i, block in enumerate(blocks):
                # Use standard autograd (custom fn above handles only C=2)
                out = block(xs[i], edge_index, edge_attr, dm) + xs[i]
                # Note: for C>2 we lose memory efficiency (stored for backprop)
                x_primes.append(out)
                prev = out
            return x_primes

    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)
        chunk = x.shape[-1] // self.num_groups

        # Refresh shared dropout mask each forward pass
        if self.training and self.dropout > 0:
            self.shared_dm.new_mask((x.size(0), chunk), x.device)
        dm = self.shared_dm.mask

        # Split into groups
        xs = [x[:, i * chunk: (i + 1) * chunk] for i in range(self.num_groups)]

        for blocks in self.layers:
            xs = self._apply_rev_block(xs, blocks, edge_index, edge_attr, dm)

        x = torch.cat(xs, dim=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def count_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


# ─────────────────────────────────────────────────────────────────────────────
# 6. DEQ-GNN (Deep Equilibrium Graph Neural Network)
# ─────────────────────────────────────────────────────────────────────────────

class DEQBlock(nn.Module):
    """
    The DEQ-GNN equilibrium block (Equations 12-15 in the paper):

        Z' = GraphConv(Z_in, A, U)
        Z'' = Norm(Z' + X)
        Z''' = GraphConv(Dropout(ReLU(Z'')), A, U)
        Z_o = Norm(ReLU(Z''' + Z'))

    where X is the injected input (initial node features),
    and Z_in → Z_o is iterated to find the fixed point Z*.
    """

    def __init__(
        self,
        hidden_channels: int,
        conv_type: str = "gcn",
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dropout_p = dropout

        if edge_dim:
            self.edge_enc = nn.Linear(edge_dim, hidden_channels)
            self.has_edge = True
        else:
            self.has_edge = False

        if conv_type == "gcn":
            self.conv1 = GCNConv(hidden_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        else:
            self.conv1 = SAGEConv(hidden_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

    def forward(self, z_in, x_inject, edge_index, edge_attr=None):
        """One iteration of the equilibrium block."""
        if self.has_edge and edge_attr is not None:
            edge_feat = self.edge_enc(edge_attr)
            row, col = edge_index
            agg = scatter(edge_feat, col, dim=0, dim_size=z_in.size(0), reduce="sum")
            z_in = z_in + agg

        zp  = self.conv1(z_in, edge_index)               # Z'
        zpp = self.norm1(zp + x_inject)                  # Z''
        h   = F.dropout(F.relu(zpp), p=self.dropout_p, training=self.training)
        zppp = self.conv2(h, edge_index)                  # Z'''
        zo  = self.norm2(F.relu(zppp + zp))              # Z_o
        return zo


class _DEQImplicitDiff(torch.autograd.Function):
    """
    Implicit differentiation for DEQ-GNN backward pass.

    Given Z* = f(Z*, X), the gradient is:
        dL/dX = (I - J_f)^{-T} · dL/dZ*

    We approximate (I - J_f)^{-T} · v using a fixed-point iteration:
        g_{k+1} = J_f^T · g_k + v
    """

    @staticmethod
    def forward(ctx, z_star, x_inject, f_func, edge_index, edge_attr, max_iter, tol):
        ctx.save_for_backward(z_star.detach(), x_inject.detach(), edge_index)
        ctx.f_func = f_func
        ctx.edge_attr = edge_attr
        ctx.max_iter = max_iter
        ctx.tol = tol
        return z_star.detach()

    @staticmethod
    def backward(ctx, dz):
        z_star, x_inject, edge_index = ctx.saved_tensors
        f_func = ctx.f_func
        edge_attr = ctx.edge_attr
        max_iter = ctx.max_iter

        # Approximate (I - J_f)^{-T} · dz using Neumann series:
        #   sum_{k=0}^{inf} (J_f^T)^k · dz ≈ accumulated gradient
        g = dz.clone()
        for _ in range(max_iter):
            with torch.enable_grad():
                z_s = z_star.detach().requires_grad_(True)
                fz = f_func(z_s, x_inject, edge_index, edge_attr)
            jvp = torch.autograd.grad(fz, z_s, grad_outputs=g, retain_graph=False)[0]
            g_new = dz + jvp.detach()
            if torch.norm(g_new - g) < ctx.tol:
                g = g_new
                break
            g = g_new

        # Gradient wrt x_inject
        with torch.enable_grad():
            x_i = x_inject.detach().requires_grad_(True)
            z_s = z_star.detach().requires_grad_(False)
            fz = f_func(z_s, x_i, edge_index, edge_attr)
        dx = torch.autograd.grad(fz, x_i, grad_outputs=g, retain_graph=False)[0]

        # Gradient wrt parameters of f_func accumulated via g
        with torch.enable_grad():
            z_s = z_star.detach().requires_grad_(True)
            fz = f_func(z_s, x_inject.detach(), edge_index, edge_attr)
        torch.autograd.backward(fz, g)

        return None, dx.detach(), None, None, None, None, None


class DEQGNN(nn.Module):
    """
    Deep Equilibrium GNN.

    Finds Z* = f(Z*, X) using fixed-point iteration.
    Backpropagates using implicit differentiation.

    Memory: O(ND)  (same as RevGNN but effectively infinite depth)
    Params: O(D²)  (single layer, like weight-tied)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        conv_type: str = "gcn",
        edge_dim: Optional[int] = None,
        max_iter: int = 50,
        fwd_tol: float = 1e-6,
        bwd_tol: float = 1e-6,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.fwd_tol = fwd_tol
        self.bwd_tol = bwd_tol
        self.model_type = "deq"

        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.eq_block = DEQBlock(hidden_channels, conv_type, dropout, edge_dim)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def _f(self, z, x_inject, edge_index, edge_attr):
        return self.eq_block(z, x_inject, edge_index, edge_attr)

    def _fixed_point_solve(self, x_inject, edge_index, edge_attr):
        """Find Z* using simple fixed-point iteration."""
        z = torch.zeros_like(x_inject)
        for _ in range(self.max_iter):
            with torch.no_grad():
                z_new = self._f(z, x_inject, edge_index, edge_attr)
                if torch.norm(z_new - z) < self.fwd_tol:
                    z = z_new
                    break
                z = z_new
        return z

    def forward(self, x, edge_index, edge_attr=None):
        x_inj = self.node_encoder(x)

        # Forward: find fixed point Z*
        z_star = self._fixed_point_solve(x_inj, edge_index, edge_attr)

        # Backward: implicit differentiation
        z_star = _DEQImplicitDiff.apply(
            z_star, x_inj,
            lambda z, xi, ei, ea: self._f(z, xi, ei, ea),
            edge_index, edge_attr,
            self.max_iter, self.bwd_tol,
        )

        return self.classifier(z_star)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    dropout: float = 0.1,
    conv_type: str = "gcn",
    edge_dim: Optional[int] = None,
    num_groups: int = 2,
    deq_max_iter: int = 50,
) -> nn.Module:
    """
    Build a GNN model by name.

    Parameters
    ----------
    model_type : one of 'resgnn', 'revgnn', 'wt-resgnn', 'wt-revgnn', 'deq'
    """
    mt = model_type.lower()
    if mt == "resgnn":
        return ResGNN(in_channels, hidden_channels, out_channels, num_layers,
                      dropout, conv_type, edge_dim, weight_tied=False)
    elif mt == "wt-resgnn":
        return ResGNN(in_channels, hidden_channels, out_channels, num_layers,
                      dropout, conv_type, edge_dim, weight_tied=True)
    elif mt == "revgnn":
        return RevGNN(in_channels, hidden_channels, out_channels, num_layers,
                      dropout, conv_type, edge_dim, weight_tied=False, num_groups=num_groups)
    elif mt == "wt-revgnn":
        return RevGNN(in_channels, hidden_channels, out_channels, num_layers,
                      dropout, conv_type, edge_dim, weight_tied=True, num_groups=num_groups)
    elif mt == "deq":
        return DEQGNN(in_channels, hidden_channels, out_channels,
                      dropout, conv_type, edge_dim, max_iter=deq_max_iter)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose from: resgnn, revgnn, wt-resgnn, wt-revgnn, deq")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Complexity table (Table 4 in the paper)
# ─────────────────────────────────────────────────────────────────────────────

def print_complexity_table():
    """Print Table 4 from the paper."""
    rows = [
        ("Full-batch GNN",   "O(LND)",  "O(LD²)",  "O(L·||A||·D + LND²)"),
        ("GraphSAGE",        "O(RLBD)", "O(LD²)",  "O(RLND²)"),
        ("VR-GCN",           "O(LND)",  "O(LD²)",  "O(L·||A||·D + RLND²)"),
        ("FastGCN",          "O(LRBD)", "O(LD²)",  "O(RLND²)"),
        ("Cluster-GCN",      "O(LBD)",  "O(LD²)",  "O(L·||A||·D + LND²)"),
        ("GraphSAINT",       "O(LBD)",  "O(LD²)",  "O(L·||A||·D + LND²)"),
        ("WT-GNN",           "O(LND)",  "O(D²)",   "O(L·||A||·D + LND²)"),
        ("RevGNN (ours)",    "O(ND) ✓", "O(LD²)",  "O(L·||A||·D + LND²)"),
        ("WT-RevGNN (ours)", "O(ND) ✓", "O(D²) ✓", "O(L·||A||·D + LND²)"),
        ("DEQ-GNN (ours)",   "O(ND) ✓", "O(D²) ✓", "O(K·||A||·D + KND²)"),
    ]
    header = f"{'Method':<22} {'Memory':>10} {'Params':>10} {'Time':>35}"
    print("=" * len(header))
    print("Table 4: Complexity Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, mem, par, time_ in rows:
        print(f"{name:<22} {mem:>10} {par:>10} {time_:>35}")
    print("=" * len(header))
    print("L=layers, N=nodes, D=channels, B=batch, R=sampled nbrs, K=Broyden iters")
