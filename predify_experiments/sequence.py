from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def reset_hebbian_state(layers: Iterable[Any]) -> None:
    """Clear short-term Hebbian state for a new trial sequence."""
    for layer in layers:
        reset = getattr(layer, "zero_boost_weight", None)
        if reset is not None:
            reset()


def run_recurrent_sequence(
    model: Any,
    inputs: Any,
    steps: int,
    *,
    hebbian_layers: Iterable[Any] = (),
    commit_update: bool = True,
    before_step: Any = None,
    after_step: Any = None,
    preserve_model_state: bool = False,
) -> list[Any]:
    """Run one input through recurrent timesteps with one deferred Hebbian commit.

    The model receives ``inputs`` on the first step and ``None`` thereafter.
    Any layer exposing ``update_enabled`` is temporarily switched to deferred
    update mode. Its pending update is committed at most once, after the final
    timestep.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    layers = tuple(hebbian_layers)
    previous_update_states: list[tuple[Any, bool]] = []
    for layer in layers:
        if hasattr(layer, "update_enabled"):
            previous_update_states.append((layer, bool(layer.update_enabled)))
            layer.update_enabled = False

    if preserve_model_state:
        if not hasattr(model, "input_mem"):
            raise ValueError("preserve_model_state requires model.input_mem")
        model.input_mem = inputs
    else:
        reset = getattr(model, "reset", None)
        if reset is not None:
            reset()

    outputs: list[Any] = []
    try:
        for timestep in range(steps):
            if before_step is not None:
                before_step(timestep)
            if preserve_model_state:
                output = model(None)
            else:
                output = model(inputs if timestep == 0 else None)
            outputs.append(output)
            if after_step is not None:
                after_step(timestep, output)

        if commit_update:
            for layer, was_enabled in previous_update_states:
                commit = getattr(layer, "commit_update", None)
                if was_enabled and commit is not None:
                    commit()
    finally:
        for layer, was_enabled in previous_update_states:
            layer.update_enabled = was_enabled

    return outputs
