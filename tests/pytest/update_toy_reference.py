"""Regenerate ``toy_model_reference.json`` from the current monolithic eliminator.

Run this script when the toy graphs are deliberately changed or when
the eliminator's expected output legitimately moves. Should be a no-op
in normal development — if running this script changes the JSON, that
is a signal to investigate before committing.

Usage:
    cd /Users/kmt/phasic
    pixi run python tests/pytest/update_toy_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toy_model import BUILDERS, THETAS


REFERENCE_PATH = Path(__file__).parent / "toy_model_reference.json"


def main() -> None:
    reference: dict[str, dict[str, dict[str, object]]] = {}
    for toy_name, builder in BUILDERS.items():
        reference[toy_name] = {}
        for theta_name, theta in THETAS.items():
            g = builder()
            g.update_weights(theta)
            ewt = [float(x) for x in np.asarray(g.expected_waiting_time())]
            exp = float(g.expectation())
            reference[toy_name][theta_name] = {
                "theta": list(theta),
                "expected_waiting_time": ewt,
                "expectation": exp,
            }
            print(
                f"{toy_name}/{theta_name}: "
                f"vertices={len(ewt)} expectation={exp!r}"
            )

    with open(REFERENCE_PATH, "w") as f:
        json.dump(reference, f, indent=2, sort_keys=True)
    print(f"\nWrote {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
