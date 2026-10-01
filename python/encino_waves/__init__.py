"""Horvath (2015): empirical directional wave spectra.

Parameters and states are values. Wave evaluation at absolute time has no
history. Foam takes an explicit previous state. Neither mutates its inputs;
tensor members are read-only by contract (PyTorch has no immutable tensors).
"""

from .model import Wave_parameters, Initial_state, Wave_frame, make_initial_state, evaluate
from .editing import Wave_basis, make_wave_basis, state_from_basis, preserve_phase, edit_state
from .foam import Foam_parameters, Foam_state, make_foam_state, step_foam

__all__ = ["Wave_parameters", "Initial_state", "Wave_frame", "make_initial_state", "evaluate",
           "Wave_basis", "make_wave_basis", "state_from_basis", "preserve_phase", "edit_state",
           "Foam_parameters", "Foam_state", "make_foam_state", "step_foam"]
