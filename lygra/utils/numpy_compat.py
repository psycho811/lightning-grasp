# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def ensure_numpy_legacy_aliases():
    """Restore NumPy aliases still used by older third-party dependencies."""
    aliases = {
        "bool": bool,
        "complex": complex,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
    }

    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
