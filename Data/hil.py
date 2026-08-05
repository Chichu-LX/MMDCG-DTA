"""Bidirectional, gated cross-hierarchy interaction learning."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


_CHECKPOINT_SUPPORTS_REENTRANT = (
    "use_reentrant" in inspect.signature(checkpoint).parameters
)


class WarpGate(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.state_projection = nn.Linear(dimension, dimension)
        self.message_projection = nn.Linear(dimension, dimension)

    def forward(self, state, message):
        gate = torch.sigmoid(
            self.state_projection(state) + self.message_projection(message)
        )
        return (1.0 - gate) * message + gate * state


class _BidirectionalHierarchyInteraction(nn.Module):
    """Apply the paper's bridge update in both information-flow directions."""

    def __init__(
        self,
        num_steps,
        dimension,
        temperature=1.0,
        negative_slope=0.2,
        use_checkpoint=True,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("attention temperature must be positive")
        self.num_steps = num_steps
        self.temperature = float(temperature)
        self.use_checkpoint = use_checkpoint
        self.message_projection = nn.Linear(dimension, dimension)
        self.warp_gate = WarpGate(dimension)
        self.bridge_gru = nn.GRUCell(dimension, dimension)
        self.node_gru = nn.GRUCell(dimension, dimension)
        self.activation = nn.LeakyReLU(negative_slope)

    @staticmethod
    def _mean(values):
        return values.mean(dim=0)

    def _one_direction(self, source, target, group_ids):
        unique_groups = torch.unique(group_ids, sorted=True)
        bridges = {
            int(group_id): self._mean(target[group_ids == group_id])
            for group_id in unique_groups
        }
        for _ in range(self.num_steps):
            updated_target = torch.zeros_like(target)
            updated_bridges = {}
            for group_id in unique_groups:
                key = int(group_id)
                mask = group_ids == group_id
                indices = mask.nonzero(as_tuple=True)[0]
                source_group = source[indices]
                target_group = target[indices]
                bridge = bridges[key]

                similarity = (
                    F.cosine_similarity(source_group, bridge.unsqueeze(0), dim=-1)
                    / self.temperature
                )
                attention = torch.softmax(similarity, dim=0).unsqueeze(-1)
                message = self.activation(
                    (attention * self.message_projection(source_group)).sum(dim=0)
                )
                gated_bridge = self.warp_gate(bridge.unsqueeze(0), message.unsqueeze(0))
                new_bridge = self.bridge_gru(
                    message.unsqueeze(0), gated_bridge
                ).squeeze(0)
                updated_bridges[key] = new_bridge

                broadcast = self.activation(self.message_projection(new_bridge))
                broadcast = broadcast.unsqueeze(0).expand_as(target_group)
                gated_target = self.warp_gate(target_group, broadcast)
                new_target = self.node_gru(gated_target, target_group)
                updated_target.index_copy_(0, indices, new_target)
            target = updated_target
            bridges = updated_bridges
        return target

    def _forward_impl(self, intra_states, inter_states, group_ids):
        inter_updated = self._one_direction(intra_states, inter_states, group_ids)
        intra_updated = self._one_direction(inter_updated, intra_states, group_ids)
        return inter_updated, intra_updated

    def forward(self, intra_states, inter_states, group_ids):
        if self.use_checkpoint and self.training and intra_states.requires_grad:
            if _CHECKPOINT_SUPPORTS_REENTRANT:
                return checkpoint(
                    self._forward_impl,
                    intra_states,
                    inter_states,
                    group_ids,
                    use_reentrant=True,
                )
            return checkpoint(self._forward_impl, intra_states, inter_states, group_ids)
        return self._forward_impl(intra_states, inter_states, group_ids)


class AtomLevelInteractiveLigand(_BidirectionalHierarchyInteraction):
    pass


class AtomLevelInteractiveProtein(_BidirectionalHierarchyInteraction):
    pass


class SubstructureLevelInteractiveLigand(_BidirectionalHierarchyInteraction):
    pass


class SubstructureLevelInteractiveProtein(_BidirectionalHierarchyInteraction):
    pass
