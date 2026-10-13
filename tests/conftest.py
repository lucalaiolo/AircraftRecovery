from pathlib import Path

import pytest

from recovery.instance import Costs, Instance, load_instance

REPO = Path(__file__).resolve().parent.parent
TOY_PATH = REPO / "examples" / "toy_network.json"
SCALES = ("tiny", "small", "medium", "large")


@pytest.fixture
def toy() -> Instance:
    return load_instance(TOY_PATH)


@pytest.fixture
def costs() -> Costs:
    """Fixed test costs, so tests do not move when config.yaml is tuned."""
    return Costs(c_delay=1.0, c_cancel_fixed=1000.0, c_cancel_pax=10.0, c_swap=500.0, c_pos=5000.0)


def micro(seed: int, kind: str) -> Instance:
    """2 aircraft, 6 flights, delay step 30: small enough to enumerate every route."""
    from recovery.generator import generate_instance
    from recovery.instance import load_config

    cfg = load_config()
    cfg["scales"]["micro"] = {"aircraft": 2, "hubs": 1, "spokes": 2, "flights": 6}
    return generate_instance("micro", seed, kind, cfg=cfg, delay_step=30)
