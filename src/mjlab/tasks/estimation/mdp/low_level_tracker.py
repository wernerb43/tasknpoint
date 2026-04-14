"""Low-level tracker action term.

Wraps a pretrained motion-tracking policy so it can drive the robot inside
the estimation environment. The term:

1. Computes the tracker's expected observation vector from the environment
   state at the start of each control step.
2. Runs the tracker network to produce raw joint-position targets.
3. Forwards those targets to an inner ``JointPositionAction`` which handles
   scale/offset/clipping and writes the actuator targets into the sim.

The high-level estimation policy communicates with the tracker *only* through
the command manager (which owns the motion choice, timing, and per-motion
target pose). That keeps the tracker interface decoupled from whatever
action space the future estimation policy will expose, so ``action_dim`` is
0 for now.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.rl import RslRlModelCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class LowLevelTrackerActionCfg(ActionTermCfg):
  """Configuration for :class:`LowLevelTrackerAction`."""

  entity_name: str = "robot"

  inner_action: JointPositionActionCfg = field(
    default_factory=lambda: JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  )
  """Inner action term used to apply scale/offset to tracker outputs and
  write the resulting joint-position targets to the sim. The shape of this
  action dictates the tracker's output dimensionality."""

  tracker_obs_group: str = "tracker"
  """Observation group containing the tracker's expected inputs, in the
  order the tracker was trained on."""

  tracker_model_cfg: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  """MLP config of the tracker policy. Must match the checkpoint's actor
  architecture (hidden dims, activation, obs normalization, distribution)."""

  checkpoint_path: str | None = None
  """Path to a local ``.pt`` tracker checkpoint. Populated by the play /
  train scripts after resolving the W&B artifact."""

  def build(self, env: ManagerBasedRlEnv) -> LowLevelTrackerAction:
    return LowLevelTrackerAction(self, env)


class LowLevelTrackerAction(ActionTerm):
  """Action term that runs a pretrained tracker policy each control step."""

  cfg: LowLevelTrackerActionCfg

  def __init__(self, cfg: LowLevelTrackerActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._inner = cfg.inner_action.build(env)

    # No high-level policy controls this term yet — motion choice/target
    # flow through the command manager. Keep the action_dim at 0 so the
    # ActionManager can still split a (num_envs, 0) policy output correctly.
    self._action_dim = 0

    # Buffer for the tracker's most recent raw (pre-scale) output. Exposed
    # via ``raw_action`` so ``mdp.last_action(action_name=...)`` returns the
    # correct "last action" observation on the next step.
    self._raw_actions = torch.zeros(
      self.num_envs, self._inner.action_dim, device=self.device
    )

    # Lazily constructed on first forward pass (we need the tracker obs
    # dim, which requires the ObservationManager to exist).
    self._tracker_model: torch.nn.Module | None = None

  # Properties.

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def inner_action(self) -> ActionTerm:
    return self._inner

  # Methods.

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._inner.reset(env_ids=env_ids)

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions  # ``action_dim`` is 0; the high-level policy doesn't feed us.

    tracker_obs = self._env.observation_manager.compute_group(
      self.cfg.tracker_obs_group
    )
    if not isinstance(tracker_obs, torch.Tensor):
      raise TypeError(
        f"Tracker obs group '{self.cfg.tracker_obs_group}' must have "
        f"concatenate_terms=True so it resolves to a single tensor."
      )

    if self._tracker_model is None:
      self._tracker_model = self._build_tracker(obs_dim=tracker_obs.shape[-1])

    with torch.inference_mode():
      tracker_output = self._tracker_model(tracker_obs)

    self._raw_actions[:] = tracker_output
    self._inner.process_actions(tracker_output)

  def apply_actions(self) -> None:
    self._inner.apply_actions()

  # Internals.

  def _build_tracker(self, obs_dim: int) -> torch.nn.Module:
    """Instantiate the tracker MLP and load weights from ``checkpoint_path``."""
    if self.cfg.checkpoint_path is None:
      raise ValueError(
        "LowLevelTrackerActionCfg.checkpoint_path is not set. Pass "
        "--tracker-wandb-run-path or --tracker-checkpoint-file to the "
        "play script so the tracker checkpoint can be resolved."
      )

    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    group = self.cfg.tracker_obs_group
    dummy_obs = TensorDict(
      {group: torch.zeros(self.num_envs, obs_dim, device=self.device)},
      batch_size=[self.num_envs],
    )
    model_cfg = self.cfg.tracker_model_cfg
    # distribution_cfg is mutated by rsl_rl (pop "class_name"), so deepcopy.
    dist_cfg = (
      copy.deepcopy(model_cfg.distribution_cfg)
      if model_cfg.distribution_cfg is not None
      else None
    )
    model = MLPModel(
      obs=dummy_obs,
      obs_groups={"actor": [group]},
      obs_set="actor",
      output_dim=self._inner.action_dim,
      hidden_dims=tuple(model_cfg.hidden_dims),
      activation=model_cfg.activation,
      obs_normalization=model_cfg.obs_normalization,
      distribution_cfg=dist_cfg,
    ).to(self.device)

    loaded = torch.load(
      self.cfg.checkpoint_path,
      map_location=self.device,
      weights_only=False,
    )
    if "actor_state_dict" not in loaded:
      raise KeyError(
        f"Tracker checkpoint '{self.cfg.checkpoint_path}' is missing "
        f"'actor_state_dict' (found keys: {sorted(loaded.keys())})."
      )
    actor_sd = dict(loaded["actor_state_dict"])
    # Match the key migration done by MjlabOnPolicyRunner.load for
    # rsl-rl 4.x → 5.x checkpoints.
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")
    model.load_state_dict(actor_sd, strict=True)
    model.eval()

    # ``as_jit`` returns a deterministic wrapper that takes a pre-concatenated
    # Tensor and already bakes in obs normalization, the MLP, and the
    # distribution's deterministic head. That's exactly the interface we want.
    inference_model = model.as_jit().to(self.device)
    inference_model.eval()
    print(
      f"[LowLevelTrackerAction] loaded tracker from {self.cfg.checkpoint_path} "
      f"(obs_dim={obs_dim}, action_dim={self._inner.action_dim})"
    )
    return inference_model
