"""Small runtime compatibility shims for pinned upstream restoration packages."""

from __future__ import annotations

import sys
import types


def ensure_basicsr_torchvision_compat() -> None:
    """Provide the legacy torchvision module imported by BasicSR 1.4.2.

    torchvision 0.18+ moved ``rgb_to_grayscale`` out of
    ``transforms.functional_tensor`` while BasicSR 1.4.2 still imports it there.
    The implementation itself remains public in ``transforms.functional``.
    """
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return
    from torchvision.transforms.functional import rgb_to_grayscale

    shim = types.ModuleType(module_name)
    shim.rgb_to_grayscale = rgb_to_grayscale
    sys.modules[module_name] = shim
