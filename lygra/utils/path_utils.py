# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

def get_package_root():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / "..").resolve()

def to_package_root(path):
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return (current_dir / ".." / "..").resolve() / path
