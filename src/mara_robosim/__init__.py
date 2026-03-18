"""MARA RoboSim: A benchmark for embodied world model learning."""

from typing import Any, Dict, Optional, Set

import gymnasium

# ---------------------------------------------------------------------------
# Environment registry
# ---------------------------------------------------------------------------

# Maps env class name → registration metadata.
ENV_CLASSES: Dict[str, Dict[str, Any]] = {}

_REGISTERED = False


def _register(
    env_id: str,
    entry_point: str,
    class_name: str,
    kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Register one gymnasium env id and record it in ENV_CLASSES."""
    gymnasium.register(
        id=env_id,
        entry_point=entry_point,
        kwargs=kwargs or {},
    )
    if class_name not in ENV_CLASSES:
        ENV_CLASSES[class_name] = {"variants": []}
    ENV_CLASSES[class_name]["variants"].append(env_id)


def register_all_environments() -> None:
    """Register every mara-robosim environment with gymnasium.

    Safe to call multiple times (idempotent).
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    _ep = "mara_robosim.gymnasium_wrapper:MARARoboSimEnv"

    # --- Single-inheritance environments ---
    _register(
        "mara/Ants-v0",
        _ep,
        "Ants",
        {"env_cls": "mara_robosim.envs.ants:PyBulletAntsEnv"},
    )
    _register(
        "mara/Balance-v0",
        _ep,
        "Balance",
        {"env_cls": "mara_robosim.envs.balance:PyBulletBalanceEnv"},
    )
    _register(
        "mara/Barrier-v0",
        _ep,
        "Barrier",
        {"env_cls": "mara_robosim.envs.barrier:PyBulletBarrierEnv"},
    )
    _register(
        "mara/Boil-v0",
        _ep,
        "Boil",
        {"env_cls": "mara_robosim.envs.boil:PyBulletBoilEnv"},
    )
    _register(
        "mara/Circuit-v0",
        _ep,
        "Circuit",
        {"env_cls": "mara_robosim.envs.circuit:PyBulletCircuitEnv"},
    )
    _register(
        "mara/Fan-v0", _ep, "Fan", {"env_cls": "mara_robosim.envs.fan:PyBulletFanEnv"}
    )
    _register(
        "mara/Float-v0",
        _ep,
        "Float",
        {"env_cls": "mara_robosim.envs.float_:PyBulletFloatEnv"},
    )
    _register(
        "mara/Grow-v0",
        _ep,
        "Grow",
        {"env_cls": "mara_robosim.envs.grow:PyBulletGrowEnv"},
    )
    _register(
        "mara/Laser-v0",
        _ep,
        "Laser",
        {"env_cls": "mara_robosim.envs.laser:PyBulletLaserEnv"},
    )
    _register(
        "mara/MagicBin-v0",
        _ep,
        "MagicBin",
        {"env_cls": "mara_robosim.envs.magic_bin:PyBulletMagicBinEnv"},
    )
    _register(
        "mara/Switch-v0",
        _ep,
        "Switch",
        {"env_cls": "mara_robosim.envs.switch:PyBulletSwitchEnv"},
    )

    # --- Multi-inheritance environments (merged) ---
    _register(
        "mara/Blocks-v0",
        _ep,
        "Blocks",
        {"env_cls": "mara_robosim.envs.blocks:PyBulletBlocksEnv"},
    )
    _register(
        "mara/Cover-v0",
        _ep,
        "Cover",
        {"env_cls": "mara_robosim.envs.cover:PyBulletCoverEnv"},
    )
    _register(
        "mara/Coffee-v0",
        _ep,
        "Coffee",
        {"env_cls": "mara_robosim.envs.coffee:PyBulletCoffeeEnv"},
    )

    # --- Domino ---
    _register(
        "mara/Domino-v0",
        _ep,
        "Domino",
        {"env_cls": "mara_robosim.envs.domino.composed_env:PyBulletDominoComposedEnv"},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make(env_id: str, **kwargs: Any) -> gymnasium.Env:
    """Create a mara-robosim gymnasium environment by id.

    Automatically calls ``register_all_environments()`` if not done yet.
    """
    register_all_environments()
    return gymnasium.make(env_id, **kwargs)


def get_all_env_ids() -> Set[str]:
    """Return the set of all registered mara-robosim environment ids."""
    register_all_environments()
    return {eid for eid in gymnasium.registry if eid.startswith("mara/")}
