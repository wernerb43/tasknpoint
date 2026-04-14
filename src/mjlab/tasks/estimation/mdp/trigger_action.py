"""Trigger action term for the estimation task.

The high-level estimation policy outputs a single scalar per env.
When the value exceeds a threshold the motion is "triggered" — the
estimation motion command reads this flag to decide when to start
the next motion clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class TriggerActionCfg(ActionTermCfg):
  """Configuration for :class:`TriggerAction`."""

  entity_name: str = "robot"

  threshold: float = 0.5
  """Value above which the trigger fires."""

  def build(self, env: ManagerBasedRlEnv) -> TriggerAction:
    return TriggerAction(self, env)


class TriggerAction(ActionTerm):
  """Single-float action that gates when the tracker starts a motion."""

  cfg: TriggerActionCfg

  def __init__(self, cfg: TriggerActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def triggered(self) -> torch.Tensor:
    """Boolean mask ``(num_envs,)`` — True where trigger > threshold."""
    return self._raw_actions[:, 0] > self.cfg.threshold

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions

  def apply_actions(self) -> None:
    pass  # No sim interaction — the motion command reads ``triggered``.

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
