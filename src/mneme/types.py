"""Shared public Protocols.

Stable interface for objects that other modules consume structurally.
Keep this module dependency-light: only stdlib + numpy. No imports from
mneme submodules to avoid cycles.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbedderProto(Protocol):
    """Anything that turns text into a unit-normalized float32 vector."""

    def embed(self, text: str) -> NDArray[np.float32]: ...
