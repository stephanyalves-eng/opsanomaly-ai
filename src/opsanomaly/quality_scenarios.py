"""Additional quality controls; old stress generator remains unchanged."""

import pandas as pd
from .stress import SCENARIOS, SENSORS, generate_stress

QUALITY_SCENARIOS = (*SCENARIOS, "legitimate_plateau", "quantized_clean", "constant_sensor")


def generate_quality_case(
    scenario: str,
    seed: int,
    rows: int = 1000,
    assets: int = 3,
    missing_probability: float = 0.1,
    start_fraction: float = 0.82,
    end_fraction: float = 0.94,
) -> pd.DataFrame:
    if scenario not in QUALITY_SCENARIOS:
        raise ValueError("Unknown quality scenario")
    source = (
        "stuck_sensor"
        if scenario == "legitimate_plateau"
        else ("clean" if scenario in {"quantized_clean", "constant_sensor"} else scenario)
    )
    frame = generate_stress(
        source, seed, rows, assets, missing_probability, start_fraction, end_fraction
    )
    if scenario == "legitimate_plateau":
        frame["label"] = 0
    elif scenario == "quantized_clean":
        held_out = frame.groupby("asset_id").cumcount() >= int(0.8 * rows)
        frame.loc[held_out, SENSORS] = frame.loc[held_out, SENSORS].round(1)
    elif scenario == "constant_sensor":
        frame["temperature"] = 50.0
    return frame
