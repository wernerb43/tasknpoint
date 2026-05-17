"""Motion set loader — reads a TOML config and produces registry strings and MotionCfg lists."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tasknpoint_project.goal_cond_tracking.mdp import MotionCfg

# Paths in [robot] sections are relative to the repo root (tasknpoint/).
# This file lives at src/tasknpoint_project/motion_sets/motion_set.py,
# so parents[4] is the repo root for editable installs.
_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class MotionSet:
    """Loaded motion set from a TOML config file."""

    train_prefix: str
    eval_prefix: str
    _entries: list[tuple[str, bool]]  # (name, enabled)
    robot_xml: str | None = field(default=None)
    """Absolute path to the robot XML, resolved from [robot].xml in the TOML."""

    @classmethod
    def from_toml(cls, path: str | Path) -> "MotionSet":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        registry = data.get("registry", {})
        entries = [(m["name"], m.get("enabled", True)) for m in data.get("motions", [])]
        robot_xml_raw = data.get("robot", {}).get("xml")
        robot_xml = None
        if robot_xml_raw is not None:
            p = Path(robot_xml_raw)
            robot_xml = str((_REPO_ROOT / p).resolve()) if not p.is_absolute() else str(p)
        return cls(
            train_prefix=registry.get("train_prefix", ""),
            eval_prefix=registry.get("eval_prefix", ""),
            _entries=entries,
            robot_xml=robot_xml,
        )

    @property
    def enabled_names(self) -> list[str]:
        return [name for name, enabled in self._entries if enabled]

    def train_registry(self) -> str:
        return ",".join(f"{self.train_prefix}/{name}" for name in self.enabled_names)

    def eval_registry(self) -> str:
        return ",".join(f"{self.eval_prefix}/{name}" for name in self.enabled_names)

    def motion_cfgs(self, lib: "dict[str, MotionCfg] | None" = None) -> "list[MotionCfg]":
        """Return MotionCfg entries for the enabled motions, in config order.

        Args:
            lib: Motion library to look up specs from. Defaults to MOTION_LIB.
        """
        if lib is None:
            from tasknpoint_project.motion_sets.motion_lib import MOTION_LIB
            lib = MOTION_LIB
        missing = [n for n in self.enabled_names if n not in lib]
        if missing:
            raise ValueError(f"Motion names not found in library: {missing}")
        return [lib[name] for name in self.enabled_names]


def filter_motion_cmd_cfg(motion_cmd_cfg: "MultiTargetMotionCommandCfg", motion_names: list[str]) -> None:
    """Trim motion_target_cfgs and motion_sampling_weights to match a subset of motions.

    When training or evaluating with fewer motions than the full library, the env cfg
    still holds all entries from MOTION_LIB. This aligns the two so _sample_motion_ids
    doesn't crash on a size mismatch between weights and loaded files.
    """
    from tasknpoint_project.motion_sets.motion_lib import MOTION_LIB
    from tasknpoint_project.goal_cond_tracking.mdp import MultiTargetMotionCommandCfg as _Cfg

    assert isinstance(motion_cmd_cfg, _Cfg)
    full_cfgs = {m.name: m for m in motion_cmd_cfg.motion_target_cfgs if m.name}
    subset = []
    for name in motion_names:
        if name in full_cfgs:
            subset.append(full_cfgs[name])
        elif name in MOTION_LIB:
            subset.append(MOTION_LIB[name])
        else:
            raise ValueError(f"No motion config found for '{name}'. Add it to motion_lib.py.")

    motion_cmd_cfg.motion_target_cfgs = subset
    motion_cmd_cfg.motion_sampling_weights = [m.sampling_weight for m in subset]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the --registry-name string for a motion set config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train registry string (default)
  uv run motion-set motion_sets/all_motions.toml

  # Eval registry string
  uv run motion-set motion_sets/all_motions.toml --eval

  # List enabled motion names
  uv run motion-set motion_sets/all_motions.toml --list
""",
    )
    parser.add_argument("config", type=Path, help="Path to motion set TOML config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval", action="store_true", help="Print eval registry string")
    mode.add_argument("--list", action="store_true", help="Print enabled motion names, one per line")
    args = parser.parse_args()

    ms = MotionSet.from_toml(args.config)

    if args.list:
        for name in ms.enabled_names:
            print(name)
    elif args.eval:
        print(ms.eval_registry())
    else:
        print(ms.train_registry())


if __name__ == "__main__":
    main()
